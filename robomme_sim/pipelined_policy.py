
from __future__ import annotations

import time

import numpy as np
import torch

from robomme_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from simplememvla.data.collator import answer_span_token_ids, subtask_span_mask
from robomme_sim.policy import RoboMMEPolicy


class PipelinedKVRunner:

    def __init__(self, vla, processor, normalize_state, num_denoising_steps,
                 max_subtask_tokens):
        self.vla = vla
        self.processor = processor
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.max_subtask_tokens = int(max_subtask_tokens)

        self.qwen = vla.backbone.model
        self.lm = self.qwen.language_model
        self.lm_head = vla.backbone.get_output_embeddings()
        self.embed = self.qwen.get_input_embeddings()
        self.dev = next(vla.parameters()).device
        self.use_proprio = bool(vla.config.use_proprio)

        self.video_token_id = int(self.qwen.config.video_token_id)
        self.image_token_id = int(self.qwen.config.image_token_id)

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise RuntimeError(
                "tokenizer encodes '\\n' as multiple ids; the pipelined newline "
                "tail assumes a single token"
            )
        self.newline_id = newline_ids[0]
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

        self.reset()

    def reset(self) -> None:
        self.a_iid = None
        self.slot_rows = None
        self._per = None
        self.prefix_cache = None
        self.prefix_keys = None
        self._t_crit: list[float] = []
        self._t_bg: list[float] = []
        self._runaway = 0

    @staticmethod
    def _flat(feat) -> torch.Tensor:
        if isinstance(feat, (list, tuple)):
            return torch.cat(list(feat), dim=0)
        if feat.dim() == 3:
            return feat.flatten(0, 1)
        return feat

    def _layout(self, prep) -> None:
        dev = self.dev
        keys = prep["video_patch_keys"]
        iid = prep["input_ids"].to(dev)
        mmtt = prep["mm_token_type_ids"].to(dev)
        vgrid = prep["video_grid_thw"].to(dev)
        img_grid = prep["image_grid_thw"].to(dev) if "image_grid_thw" in prep else None
        pos, deltas = self.qwen.get_rope_index(
            iid, image_grid_thw=img_grid, video_grid_thw=vgrid,
            attention_mask=torch.ones_like(iid), mm_token_type_ids=mmtt)
        vpos = (iid[0] == self.video_token_id).nonzero().squeeze(-1)
        n_slots = len(keys)
        if n_slots < 2:
            raise RuntimeError(f"pipelined needs >= 2 temporal patches, got {n_slots}")
        if vpos.numel() % n_slots:
            raise RuntimeError(
                f"{vpos.numel()} video tokens do not tile into {n_slots} patches")
        self._per = vpos.numel() // n_slots
        self.slot_rows = [vpos[self._per * s: self._per * (s + 1)]
                          for s in range(n_slots)]
        self.a_iid, self.a_pos, self.a_mmtt = iid, pos, mmtt
        self.a_vgrid, self.a_deltas = vgrid, deltas

    def _build_prefix(self, keys_head, S1, feat_cache) -> None:
        emb = self.embed(self.a_iid[:, :S1])
        vmask = self.a_iid[0, :S1] == self.video_token_id
        emb[0][vmask] = torch.cat([feat_cache[k] for k in keys_head], dim=0).to(emb.dtype)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=self.a_pos[:, :, :S1], use_cache=True)
        self.prefix_cache = out.past_key_values
        self.prefix_keys = list(keys_head)

    @torch.no_grad()
    def step(self, prep, state_norm, seed):
        dev = self.dev
        pol = prep["policy"]
        keys = prep["video_patch_keys"]

        gh, gw = pol._prompt_cache["grid_hw"]
        miss = [(k, px) for k, px in prep["video_patch_pixels"].items()
                if k not in pol._patch_feat_cache]
        if miss:
            rows = torch.cat([px for _, px in miss], 0).to(dev)
            grid = torch.tensor([[1, gh, gw]] * len(miss), device=dev, dtype=torch.long)
            feats = self.qwen.get_video_features(rows, grid, return_dict=True).pooler_output
            for (k, _), f in zip(miss, feats):
                pol._patch_feat_cache[k] = f

        first = self.a_iid is None
        if first:
            self._layout(prep)
        elif not torch.equal(prep["input_ids"].to(dev), self.a_iid):
            raise RuntimeError("prompt changed mid-episode; pipelined needs a frozen prompt")
        n_slots = len(self.slot_rows)
        S1 = int(self.slot_rows[n_slots - 2][-1]) + 1
        if first:
            L = int(self.a_iid.shape[1])
            print(f"[pipeline] prompt={L} tokens -> background={S1} (slots 0..{n_slots - 2}, "
                  f"{100.0 * S1 / L:.0f}%), critical={L - S1} (last patch + wrist + "
                  "instruction) + sub-task decode + DiT", flush=True)

        torch.cuda.synchronize()
        t0 = time.time()
        if self.prefix_cache is None or self.prefix_keys != list(keys[:-1]):
            self._build_prefix(list(keys[:-1]), S1, pol._patch_feat_cache)
        cache = self.prefix_cache
        self.prefix_cache = None

        iid, pos = self.a_iid, self.a_pos
        emb = self.embed(iid[:, S1:])
        span = iid[0, S1:]
        vmask = span == self.video_token_id
        if int(vmask.sum()) != self._per:
            raise RuntimeError("the fed span must hold exactly one temporal patch")
        emb[0][vmask] = pol._patch_feat_cache[keys[-1]].to(emb.dtype)
        img_grid = prep["image_grid_thw"].to(dev) if "image_grid_thw" in prep else None
        if img_grid is not None:
            ifeat = self._flat(self.qwen.get_image_features(
                prep["pixel_values"].to(dev), img_grid, return_dict=True).pooler_output)
            imask = span == self.image_token_id
            if int(imask.sum()) != ifeat.shape[0]:
                raise RuntimeError("wrist image tokens and ViT features disagree")
            emb[0][imask] = ifeat.to(emb.dtype)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=pos[:, :, S1:], past_key_values=cache, use_cache=True)
        result = self._decode_and_act(iid, img_grid, out, out.past_key_values,
                                      state_norm, seed)
        torch.cuda.synchronize()
        t1 = time.time()

        self._build_prefix(list(keys[1:]), S1, pol._patch_feat_cache)
        torch.cuda.synchronize()
        t2 = time.time()

        self._t_crit.append(t1 - t0)
        self._t_bg.append(t2 - t1)
        if len(self._t_crit) % 25 == 0:
            import statistics as st
            print(f"[pipeline] n={len(self._t_crit)} "
                  f"critical={st.mean(self._t_crit):.3f}s "
                  f"background={st.mean(self._t_bg):.3f}s "
                  f"(deployment latency = critical; 1.0 s budget for background)",
                  flush=True)
        return result

    def _decode_and_act(self, iid, img_grid, prefill_out, cache, state_norm, seed):
        dev = self.dev
        L = iid.shape[1]
        max_new = self.max_subtask_tokens
        next_tok = self.lm_head(prefill_out.last_hidden_state[:, -1]).float().argmax(-1)
        sampled = [int(next_tok[0])]
        eos_seen = sampled[0] == self.im_end_id
        newline_fed = False
        runaway = False
        dec_hidden = []
        fed = 0
        delta = self.a_deltas[:, 0].to(dev)

        def done(steps_done: int) -> bool:
            if eos_seen:
                return newline_fed and steps_done >= len(sampled) + 1
            return len(sampled) >= max_new and steps_done >= len(sampled)

        for step in range(max_new + 2):
            if step < len(sampled):
                tok = sampled[step]
            elif eos_seen and not newline_fed:
                tok = self.newline_id
                newline_fed = True
            else:
                break
            fed += 1
            p = (delta + L + fed - 1).view(1, 1, 1).expand(3, 1, 1)
            out = self.lm(input_ids=torch.tensor([[tok]], device=dev),
                          attention_mask=None, position_ids=p,
                          past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
            dec_hidden.append(h)
            nxt = int(self.lm_head(h).float().argmax(-1)[0])
            if not eos_seen and step == len(sampled) - 1:
                if len(sampled) < max_new:
                    sampled.append(nxt)
                    if nxt == self.im_end_id:
                        eos_seen = True
                else:
                    sampled.append(self.im_end_id)
                    eos_seen = True
                    runaway = True
            if done(step + 1):
                break

        if runaway:
            self._runaway += 1
            print(f"[pipeline] WARNING: sub-task decode hit the {max_new}-token budget "
                  "without emitting <|im_end|>; forced a terminator so the DiT condition "
                  "keeps its trained shape, but this decision is conditioned on a "
                  "degenerate sub-task.", flush=True)

        gen_ids = torch.tensor(sampled, dtype=iid.dtype, device=dev)
        full = torch.cat([iid[0], gen_ids])
        if int(full[-1]) == self.im_end_id:
            full = torch.cat([full, torch.tensor([self.newline_id], device=dev,
                                                 dtype=full.dtype)])
        gen_len = full.shape[0] - L
        fiid = full[None]
        fatt = torch.ones_like(fiid)
        fmmtt = torch.cat([self.a_mmtt[0],
                           torch.zeros(gen_len, dtype=self.a_mmtt.dtype, device=dev)])[None]
        span = subtask_span_mask(full, self.think_close_id, self.whitespace_ids)[None].to(dev)
        mm_full = {"image_grid_thw": img_grid, "video_grid_thw": self.a_vgrid,
                   "mm_token_type_ids": fmmtt}
        pos_full = self.vla._position_ids(fiid, fatt, mm_full)
        hid = torch.stack(dec_hidden, 1)
        hidden_full = hid.new_zeros(1, fiid.shape[1], hid.shape[-1])
        if gen_len > hid.shape[1]:
            raise RuntimeError("decode hidden capture shorter than generated span")
        hidden_full[0, L: L + gen_len] = hid[0, :gen_len]
        cond_hidden, cond_token, cond_mask, cond_pos = self.vla._gather_condition(
            hidden_states=hidden_full, input_ids=fiid, span=span,
            position_ids=pos_full, attention_mask=fatt)
        torch.manual_seed(seed)
        actions = self.vla.action_head.sample(
            cond_hidden, condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask, condition_token_embeds=cond_token,
            num_steps=self.num_denoising_steps, temperature=1.0,
            state=state_norm if self.use_proprio else None)
        subtask = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return subtask, actions


class PipelinedBufferPolicy(RoboMMEPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_runner: PipelinedKVRunner | None = None

    def reset(self) -> None:
        super().reset()
        if self._stream_runner is not None:
            self._stream_runner.reset()


class PipelinedEvalGroup:

    def __init__(self, model, processor, unnormalize_action, normalize_state,
                 num_denoising_steps: int = 10, eval_temperature: float = 1.0,
                 max_subtask_tokens: int = 64, device: torch.device | None = None):
        if float(eval_temperature) != 1.0:
            raise ValueError("the pipelined path supports eval_temperature=1.0 only")
        self.model = model
        self.processor = processor
        self.unnormalize = unnormalize_action
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.device = device or next(model.parameters()).device
        self.action_dim = int(model.config.action_dim)

    def _runner_for(self, pol: PipelinedBufferPolicy) -> PipelinedKVRunner:
        if pol._stream_runner is None:
            pol._stream_runner = PipelinedKVRunner(
                self.model, self.processor, self.normalize_state,
                self.num_denoising_steps, self.max_subtask_tokens)
        return pol._stream_runner

    @torch.no_grad()
    def generate_batch(self, processed_list, state_norm_list):
        self.model.eval()
        results = []
        for prep, state in zip(processed_list, state_norm_list):
            pol: PipelinedBufferPolicy = prep["policy"]
            runner = self._runner_for(pol)
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            subtask, actions = runner.step(prep, state, seed)
            norm = actions[0, :, : self.action_dim].float().cpu()
            unnorm = self.unnormalize.unnormalize(norm).numpy().astype(np.float32)
            results.append((unnorm, subtask))
        return results
