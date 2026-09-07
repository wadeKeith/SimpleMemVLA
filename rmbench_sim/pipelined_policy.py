
from __future__ import annotations

import time

import numpy as np
import torch

from rmbench_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from simplememvla.data.collator import answer_span_token_ids, subtask_span_mask
from rmbench_sim.policy import RMBenchPolicy


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

        self.video_token_id = int(self.qwen.config.video_token_id)
        self.image_token_id = int(self.qwen.config.image_token_id)

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

        self._t_crit: list[float] = []
        self._t_bg: list[float] = []
        self._hits = 0
        self._misses = 0
        self._runaways = 0
        self.reset()

    def reset(self) -> None:
        self.a_iid = None
        self.a_pos = None
        self.a_mmtt = None
        self.a_vgrid = None
        self.a_deltas = None
        self.slot_rows = None
        self._per = None
        self.prefix_cache = None
        self.prefix_keys = None
        self.prefix_ids = None
        self.prefix_pos = None

    @staticmethod
    def _flat(feat) -> torch.Tensor:
        if isinstance(feat, (list, tuple)):
            return torch.cat(list(feat), dim=0)
        if feat.dim() == 3:
            return feat.flatten(0, 1)
        return feat

    def _layout(self, prep, keys) -> None:
        dev = self.dev
        iid = prep["input_ids"].to(dev)
        mmtt = prep["mm_token_type_ids"].to(dev)
        vgrid = prep["video_grid_thw"].to(dev)
        img_grid = prep["image_grid_thw"].to(dev)
        pos, deltas = self.qwen.get_rope_index(
            iid, image_grid_thw=img_grid, video_grid_thw=vgrid,
            attention_mask=torch.ones_like(iid), mm_token_type_ids=mmtt)
        vpos = (iid[0] == self.video_token_id).nonzero().squeeze(-1)
        n_slots = len(keys)
        if vpos.numel() % n_slots:
            raise RuntimeError(
                f"{vpos.numel()} video tokens do not tile into {n_slots} patches")
        self._per = vpos.numel() // n_slots
        self.slot_rows = [vpos[self._per * s: self._per * (s + 1)]
                          for s in range(n_slots)]
        self.a_iid, self.a_pos, self.a_mmtt = iid, pos, mmtt
        self.a_vgrid, self.a_deltas = vgrid, deltas

    def _build_prefix(self, keys_head, S1, feat_cache) -> None:
        if S1 <= 0:
            self.prefix_cache = self.prefix_keys = None
            self.prefix_ids = self.prefix_pos = None
            return
        iid, pos = self.a_iid, self.a_pos
        emb = self.embed(iid[:, :S1])
        vmask = iid[0, :S1] == self.video_token_id
        feats = torch.cat([feat_cache[k] for k in keys_head], dim=0)
        if int(vmask.sum()) != feats.shape[0]:
            raise RuntimeError(
                f"prefix holds {int(vmask.sum())} video tokens but "
                f"{len(keys_head)} cached patches supply {feats.shape[0]}")
        emb[0][vmask] = feats.to(emb.dtype)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=pos[:, :, :S1], use_cache=True)
        self.prefix_cache = out.past_key_values
        self.prefix_keys = list(keys_head)
        self.prefix_ids = iid[:, :S1].clone()
        self.prefix_pos = pos[:, :, :S1].clone()

    @torch.no_grad()
    def step(self, prep, state_norm, seed):
        dev = self.dev
        pol = prep["policy"]
        keys = prep["video_patch_keys"]
        feat_cache = pol._patch_feat_cache
        n_slots = len(keys)
        cap = pol.n_frames // 2

        gh, gw = pol._prompt_cache["grid_hw"]
        miss = [(k, px) for k, px in prep["video_patch_pixels"].items()
                if k not in pol._patch_feat_cache]
        if miss:
            rows = torch.cat([px for _, px in miss], 0).to(dev)
            grid = torch.tensor([[1, gh, gw]] * len(miss), device=dev, dtype=torch.long)
            feats = self.qwen.get_video_features(rows, grid, return_dict=True).pooler_output
            for (k, _), f in zip(miss, feats):
                pol._patch_feat_cache[k] = f

        iid_now = prep["input_ids"].to(dev)
        if self.a_iid is None or not torch.equal(iid_now, self.a_iid):
            self._layout(prep, keys)
        iid, pos = self.a_iid, self.a_pos
        S1 = int(self.slot_rows[n_slots - 2][-1]) + 1 if n_slots >= 2 else 0

        torch.cuda.synchronize()
        t0 = time.time()
        usable = (
            self.prefix_cache is not None
            and S1 > 0
            and self.prefix_keys == list(keys[:-1])
            and self.prefix_ids.shape[1] == S1
            and torch.equal(self.prefix_ids, iid[:, :S1])
            and torch.equal(self.prefix_pos, pos[:, :, :S1])
        )
        if usable:
            self._hits += 1
        else:
            if S1 > 0:
                self._misses += 1
            self._build_prefix(list(keys[:-1]), S1, feat_cache)
        cache = self.prefix_cache
        self.prefix_cache = None

        emb = self.embed(iid[:, S1:])
        span = iid[0, S1:]
        vmask = span == self.video_token_id
        fed_keys = keys if S1 == 0 else keys[-1:]
        vfeat = torch.cat([feat_cache[k] for k in fed_keys], dim=0)
        if int(vmask.sum()) != vfeat.shape[0]:
            raise RuntimeError(
                f"critical span holds {int(vmask.sum())} video tokens but the fed "
                f"patches supply {vfeat.shape[0]}")
        emb[0][vmask] = vfeat.to(emb.dtype)
        img_grid = prep["image_grid_thw"].to(dev)
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

        if n_slots < cap:
            self._build_prefix(list(keys), int(self.slot_rows[n_slots - 1][-1]) + 1,
                               feat_cache)
        else:
            self._build_prefix(list(keys[1:]), S1, feat_cache)
        torch.cuda.synchronize()
        t2 = time.time()

        self._t_crit.append(t1 - t0)
        self._t_bg.append(t2 - t1)
        if len(self._t_crit) % 25 == 0:
            import statistics as st

            n = len(self._t_crit)
            print(f"[pipeline] n={n} "
                  f"critical={st.mean(self._t_crit[-25:]):.3f}s "
                  f"background={st.mean(self._t_bg[-25:]):.3f}s "
                  f"prefetch_hits={self._hits}/{self._hits + self._misses} "
                  f"(deployment latency = critical; 0.96 s budget for background)",
                  flush=True)
        return result

    def _decode_and_act(self, iid, img_grid, prefill_out, cache, state_norm, seed):
        dev = self.dev
        L = iid.shape[1]
        next_tok = self.lm_head(prefill_out.last_hidden_state[:, -1]).float().argmax(-1)
        sampled = [int(next_tok[0])]
        eos_seen = sampled[0] == self.im_end_id
        newline_fed = False
        runaway = False
        dec_hidden = []
        fed = 0
        delta = self.a_deltas[:, 0].to(dev)
        for step in range(self.max_subtask_tokens + 2):
            if step < len(sampled):
                tok = sampled[step]
            elif eos_seen and not newline_fed and self.newline_id is not None:
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
            if eos_seen or step != len(sampled) - 1:
                continue
            if len(sampled) < self.max_subtask_tokens:
                sampled.append(nxt)
                if nxt == self.im_end_id:
                    eos_seen = True
            else:
                sampled.append(self.im_end_id)
                eos_seen = True
                runaway = True
        if runaway:
            self._runaways += 1
            print(f"[eval] WARNING: sub-task decode hit the {self.max_subtask_tokens}"
                  "-token budget without emitting <|im_end|>; forced a terminator so "
                  "the DiT condition keeps its trained shape, but this decision is "
                  "conditioned on a degenerate sub-task.", flush=True)

        gen_ids = torch.tensor(sampled, dtype=iid.dtype, device=dev)
        full = torch.cat([iid[0], gen_ids])
        if self.newline_id is not None and int(full[-1]) == self.im_end_id:
            full = torch.cat([full, torch.tensor([self.newline_id], device=dev,
                                                 dtype=full.dtype)])
        gen_len = full.shape[0] - L
        fiid = full[None]
        fatt = torch.ones_like(fiid)
        fmmtt = torch.cat([self.a_mmtt[0],
                           torch.zeros(gen_len, dtype=self.a_mmtt.dtype, device=dev)])[None]
        span = subtask_span_mask(full, self.think_close_id, self.whitespace_ids)[None].to(dev)
        if bool(span[0, :L].any()):
            raise RuntimeError(
                "subtask span leaked into the prompt region (template/tokenizer "
                "drift?) — conditioning would silently read zeros")
        mm_full = {"image_grid_thw": img_grid, "video_grid_thw": self.a_vgrid,
                   "mm_token_type_ids": fmmtt}
        pos_full = self.vla._position_ids(fiid, fatt, mm_full)
        hid = torch.stack(dec_hidden, 1)
        if gen_len > hid.shape[1]:
            raise RuntimeError("decode hidden capture shorter than generated span")
        hidden_full = hid.new_zeros(1, fiid.shape[1], hid.shape[-1])
        hidden_full[0, L: L + gen_len] = hid[0, :gen_len]
        cond_hidden, cond_token, cond_mask, cond_pos = self.vla._gather_condition(
            hidden_states=hidden_full, input_ids=fiid, span=span,
            position_ids=pos_full, attention_mask=fatt)
        torch.manual_seed(seed)
        actions = self.vla.action_head.sample(
            cond_hidden, condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask, condition_token_embeds=cond_token,
            num_steps=self.num_denoising_steps, temperature=1.0, state=state_norm)
        subtask = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return subtask, actions


class PipelinedBufferPolicy(RMBenchPolicy):

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
