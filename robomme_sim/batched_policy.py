
from __future__ import annotations

import numpy as np
import torch

from robomme_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()


class BatchedEvalPolicy:

    def __init__(
        self,
        model,
        processor,
        unnormalize_action,
        normalize_state,
        num_denoising_steps: int = 10,
        eval_temperature: float = 1.0,
        max_subtask_tokens: int = 64,
        device: torch.device | None = None,
    ):
        self.model = model
        self.processor = processor
        self.unnormalize = unnormalize_action
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.eval_temperature = float(eval_temperature)
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.device = device or next(model.parameters()).device

        cfg = model.config
        self.use_proprio = bool(cfg.use_proprio)
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)

        from simplememvla.data.collator import answer_span_token_ids

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else self.im_end_id

    def _position_ids(self, input_ids, attention_mask, mm_kwargs):
        return self.model._position_ids(input_ids, attention_mask, mm_kwargs)

    def _gather_condition(self, hidden_states, input_ids, span, position_ids, attention_mask):
        return self.model._gather_condition(
            hidden_states=hidden_states,
            input_ids=input_ids,
            span=span,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        processed_list: list[dict],
        state_norm_list: list,
    ) -> list[tuple[np.ndarray, str]]:
        from simplememvla.data.collator import subtask_span_mask

        self.model.eval()
        N = len(processed_list)
        dev = self.device
        tok = self.processor.tokenizer
        pad = self.pad_token_id

        qwen = self.model.backbone.model
        lm = qwen.language_model
        lm_head = self.model.backbone.get_output_embeddings()
        embed = qwen.get_input_embeddings()
        video_token_id = int(qwen.config.video_token_id)
        image_token_id = int(qwen.config.image_token_id)

        miss: list[tuple[object, tuple, torch.Tensor]] = []
        for p in processed_list:
            pol = p["policy"]
            for k, rows in p["video_patch_pixels"].items():
                if k not in pol._patch_feat_cache:
                    miss.append((pol, k, rows))
        if miss:
            gh, gw = processed_list[0]["policy"]._prompt_cache["grid_hw"]
            pixels = torch.cat([rows for _, _, rows in miss], dim=0).to(dev)
            grid = torch.tensor([[1, gh, gw]] * len(miss), device=dev, dtype=torch.long)
            feats = qwen.get_video_features(pixels, grid, return_dict=True).pooler_output
            for (pol, k, _), f in zip(miss, feats):
                pol._patch_feat_cache[k] = f

        has_images = "pixel_values" in processed_list[0]
        img_feats_per_env: list[torch.Tensor | None] = [None] * N
        img_grids_cat = None
        if has_images:
            assert all("pixel_values" in p for p in processed_list), \
                "image channel must be present for every env"
            pix = torch.cat([p["pixel_values"] for p in processed_list], dim=0).to(dev)
            img_grids_cat = torch.cat(
                [p["image_grid_thw"] for p in processed_list], dim=0
            ).to(dev)
            feats = qwen.get_image_features(pix, img_grids_cat, return_dict=True).pooler_output
            idx = 0
            for i, p in enumerate(processed_list):
                n_img = p["image_grid_thw"].shape[0]
                img_feats_per_env[i] = torch.cat(feats[idx: idx + n_img], dim=0)
                idx += n_img

        maxP = max(p["input_ids"].shape[1] for p in processed_list)
        iid = torch.full((N, maxP), pad, dtype=torch.long, device=dev)
        att = torch.zeros((N, maxP), dtype=torch.long, device=dev)
        mmtt = torch.zeros((N, maxP), dtype=torch.long, device=dev)
        prompt_lens = []
        for i, p in enumerate(processed_list):
            L = p["input_ids"].shape[1]
            iid[i, maxP - L:] = p["input_ids"][0].to(dev)
            att[i, maxP - L:] = 1
            mmtt[i, maxP - L:] = p["mm_token_type_ids"][0].to(dev)
            prompt_lens.append(L)
        emb = embed(iid)
        for i, p in enumerate(processed_list):
            pol = p["policy"]
            vid_feat = torch.cat(
                [pol._patch_feat_cache[k] for k in p["video_patch_keys"]], dim=0
            )
            vmask = iid[i] == video_token_id
            if int(vmask.sum()) != vid_feat.shape[0]:
                raise RuntimeError(
                    f"video token count {int(vmask.sum())} != cached features "
                    f"{vid_feat.shape[0]} (env {i})"
                )
            emb[i][vmask] = vid_feat.to(emb.dtype)
            if has_images:
                imask = iid[i] == image_token_id
                if int(imask.sum()) != img_feats_per_env[i].shape[0]:
                    raise RuntimeError("image token count != wrist features")
                emb[i][imask] = img_feats_per_env[i].to(emb.dtype)

        grids_cat = torch.cat([p["video_grid_thw"] for p in processed_list], dim=0).to(dev)
        position_ids, rope_deltas = qwen.get_rope_index(
            iid,
            image_grid_thw=img_grids_cat,
            video_grid_thw=grids_cat,
            attention_mask=att,
            mm_token_type_ids=mmtt,
        )

        out = lm(
            inputs_embeds=emb,
            attention_mask=att,
            position_ids=position_ids,
            use_cache=True,
        )
        cache = out.past_key_values
        next_tok = lm_head(out.last_hidden_state[:, -1]).float().argmax(-1)

        max_new = self.max_subtask_tokens
        sampled: list[list[int]] = [[int(next_tok[i])] for i in range(N)]
        eos_seen = [sampled[i][0] == self.im_end_id for i in range(N)]
        newline_fed = [False] * N
        runaway = [False] * N
        dec_hidden: list[torch.Tensor] = []
        cur_att = att
        valid_len = att.sum(-1)
        delta = rope_deltas[:, 0].to(dev)

        def row_done(i: int, steps_done: int) -> bool:
            if eos_seen[i]:
                return newline_fed[i] and steps_done >= len(sampled[i]) + 1
            return len(sampled[i]) >= max_new and steps_done >= len(sampled[i])

        for step in range(max_new + 2):
            feed = torch.full((N, 1), pad, dtype=torch.long, device=dev)
            for i in range(N):
                if step < len(sampled[i]):
                    feed[i, 0] = sampled[i][step]
                elif eos_seen[i] and not newline_fed[i] and self.newline_id is not None:
                    feed[i, 0] = self.newline_id
                    newline_fed[i] = True
            cur_att = torch.cat(
                [cur_att, torch.ones((N, 1), dtype=cur_att.dtype, device=dev)], dim=-1
            )
            valid_len = valid_len + 1
            pos = (delta + valid_len - 1).view(1, N, 1).expand(3, N, 1)
            out = lm(
                input_ids=feed,
                attention_mask=cur_att,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
            dec_hidden.append(h)
            nxt = lm_head(h).float().argmax(-1)
            for i in range(N):
                if eos_seen[i] or step != len(sampled[i]) - 1:
                    continue
                if len(sampled[i]) < max_new:
                    t = int(nxt[i])
                    sampled[i].append(t)
                    if t == self.im_end_id:
                        eos_seen[i] = True
                else:
                    sampled[i].append(self.im_end_id)
                    eos_seen[i] = True
                    runaway[i] = True
            if all(row_done(i, step + 1) for i in range(N)):
                break

        if any(runaway):
            print(
                f"[eval] WARNING: {sum(runaway)}/{N} row(s) hit the "
                f"{max_new}-token sub-task budget without emitting <|im_end|>; "
                "forced a terminator so the DiT condition keeps its trained "
                "shape, but those decisions are conditioned on a degenerate "
                "sub-task.",
                flush=True,
            )

        full_ids_list, mmtt_full_list, span_list, subtask_text_list = [], [], [], []
        gen_lens = []
        for i in range(N):
            prompt_ids = processed_list[i]["input_ids"][0].to(dev)
            gen_ids = torch.tensor(sampled[i], dtype=prompt_ids.dtype, device=dev)
            full = torch.cat([prompt_ids, gen_ids])
            if (self.newline_id is not None and gen_ids.numel() > 0
                    and int(full[-1]) == self.im_end_id):
                full = torch.cat(
                    [full, torch.tensor([self.newline_id], device=dev, dtype=full.dtype)]
                )
            full_ids_list.append(full)
            gen_lens.append(full.shape[0] - prompt_lens[i])
            mmtt_full_list.append(torch.cat([
                processed_list[i]["mm_token_type_ids"][0].to(dev),
                torch.zeros(full.shape[0] - prompt_lens[i], dtype=mmtt.dtype, device=dev),
            ]))
            span_list.append(
                subtask_span_mask(full, self.think_close_id, self.whitespace_ids).to(dev)
            )
            subtask_text_list.append(tok.decode(gen_ids, skip_special_tokens=True).strip())

        maxF = max(f.shape[0] for f in full_ids_list)
        fiid = torch.full((N, maxF), pad, dtype=torch.long, device=dev)
        fatt = torch.zeros((N, maxF), dtype=torch.long, device=dev)
        fmmtt = torch.zeros((N, maxF), dtype=torch.long, device=dev)
        fspan = torch.zeros((N, maxF), dtype=torch.bool, device=dev)
        for i in range(N):
            Lf = full_ids_list[i].shape[0]
            fiid[i, :Lf] = full_ids_list[i]
            fatt[i, :Lf] = 1
            fmmtt[i, :Lf] = mmtt_full_list[i]
            fspan[i, :Lf] = span_list[i]
            if bool(span_list[i][: prompt_lens[i]].any()):
                raise RuntimeError(
                    "subtask span leaked into the prompt region (template/tokenizer "
                    "drift?) — batched conditioning would silently read zeros"
                )
        mm_full = {"image_grid_thw": img_grids_cat, "video_grid_thw": grids_cat,
                   "mm_token_type_ids": fmmtt}
        position_ids_full = self._position_ids(fiid, fatt, mm_full)

        dec_stack = torch.stack(dec_hidden, dim=1)
        hidden_full = dec_stack.new_zeros(N, maxF, dec_stack.shape[-1])
        for i in range(N):
            L, g = prompt_lens[i], gen_lens[i]
            if g > dec_stack.shape[1]:
                raise RuntimeError("decode hidden capture shorter than generated span")
            hidden_full[i, L: L + g] = dec_stack[i, :g]

        cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
            hidden_states=hidden_full, input_ids=fiid, span=fspan,
            position_ids=position_ids_full, attention_mask=fatt,
        )

        state_batch = None
        if self.use_proprio:
            state_batch = torch.stack(
                [s[0] if s.dim() == 2 else s for s in state_norm_list]
            ).to(dev)
        actions_b = self.model.action_head.sample(
            cond_hidden,
            condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask,
            condition_token_embeds=cond_token,
            num_steps=self.num_denoising_steps,
            temperature=self.eval_temperature,
            state=state_batch,
        )

        results = []
        for i in range(N):
            norm_a = actions_b[i, :, : self.action_dim].float().cpu()
            actions_unnorm = self.unnormalize.unnormalize(norm_a).numpy().astype(np.float32)
            results.append((actions_unnorm, subtask_text_list[i]))
        return results
