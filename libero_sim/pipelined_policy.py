
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from libero_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from simplememvla.data.collator import answer_span_token_ids, subtask_span_mask
from simplememvla.data.messages import variable_history_frames
from libero_sim.policy import LiberoPolicy


class PipelinedBufferPolicy(LiberoPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_runner: PipelinedKVRunner | None = None
        self._render_clip_frames: int | None = None

    def reset(self) -> None:
        super().reset()
        self._render_clip_frames = None
        if self._stream_runner is not None:
            self._stream_runner.reset()

    def next_window(self, horizon: int) -> tuple[int, list[tuple]] | None:
        keys: list[tuple] = []
        n_frames_out: int | None = None
        for cam in self.history_image_keys:
            buf = self._buffers[cam]
            if not buf:
                return None
            base = self._frames_seen[cam] - len(buf)
            length = len(buf) + int(horizon)
            cap = self._buffer_caps[cam]
            if length > cap:
                base += length - cap
                length = cap
            cur = length - 1
            n, stride = self.n_frames, self.stride
            if self.variable_history:
                m = variable_history_frames(min(cur // stride + 1, n))
                pos = [max(0, cur - (m - 1 - i) * stride) for i in range(m)]
            else:
                m = n
                offsets = [-(n - 1 - i) * stride for i in range(n)] if n > 1 else [0]
                pos = [max(0, cur + off) for off in offsets]
            if len(pos) % 2:
                return None
            gidx = [base + p for p in pos]
            tag = cam.split(".")[-1]
            for j in range(0, len(gidx), 2):
                keys.append((tag, gidx[j], gidx[j + 1]))
            if n_frames_out is None:
                n_frames_out = len(gidx)
            elif n_frames_out != len(gidx):
                raise RuntimeError("history cameras disagree on clip length")
        if n_frames_out is None:
            return None
        return n_frames_out, keys

    def _visual_inputs(self):
        if self._render_clip_frames is None:
            return super()._visual_inputs()
        n = int(self._render_clip_frames)
        sampled_fps = self.native_fps / self.stride
        history = set(self.history_image_keys)
        videos, metas, images = [], [], []
        for key in self.image_keys:
            buf = self._buffers[key]
            if not buf:
                raise RuntimeError("call observe() before predict()")
            if key in history:
                videos.append(np.repeat(np.asarray(buf[-1])[None], n, axis=0))
                metas.append(
                    VideoMetadata(
                        total_num_frames=n,
                        fps=sampled_fps,
                        frames_indices=list(range(n)),
                        duration=n / sampled_fps,
                        video_backend="rollout",
                    )
                )
            else:
                images.append(np.asarray(buf[-1]))
        return videos, metas, images

    def _render_layout(self, prompt: str, clip_frames: int) -> dict:
        self._render_clip_frames = int(clip_frames)
        try:
            return self._process(prompt, add_generation_prompt=True)
        finally:
            self._render_clip_frames = None


class PipelinedKVRunner:

    def __init__(self, vla, processor, normalize_state, num_denoising_steps,
                 max_subtask_tokens, eval_temperature, execute_horizon):
        self.vla = vla
        self.processor = processor
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.eval_temperature = float(eval_temperature)
        self.execute_horizon = int(execute_horizon)

        self.qwen = vla.backbone.model
        self.lm = self.qwen.language_model
        self.lm_head = vla.backbone.get_output_embeddings()
        self.embed = self.qwen.get_input_embeddings()
        self.dev = next(vla.parameters()).device

        self.video_token_id = int(self.qwen.config.video_token_id)
        self.image_token_id = int(self.qwen.config.image_token_id)

        tok = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(tok)
        think_open_id = tok.convert_tokens_to_ids("<think>")
        self.suppress_token_ids = [
            i for i in (think_open_id, self.think_close_id)
            if isinstance(i, int) and i >= 0
        ]
        newline_ids = tok.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

        self.reset()

    def reset(self) -> None:
        self._layouts: dict[int, dict] = {}
        self.prefix_cache = None
        self.prefix_keys: list[tuple] | None = None
        self.prefix_layout: dict | None = None
        self.t_crit: list[float] = []
        self.t_bg: list[float] = []
        self.n_hit = 0
        self.n_miss = 0
        self.n_nosplit = 0
        self.n_runaway = 0
        self._warned_layout = False

    @staticmethod
    def _flat(feat) -> torch.Tensor:
        if isinstance(feat, (list, tuple)):
            return torch.cat(list(feat), dim=0)
        if feat.dim() == 3:
            return feat.flatten(0, 1)
        return feat

    def _make_layout(self, iid, mmtt, vgrid, img_grid, n_slots: int) -> dict:
        pos, deltas = self.qwen.get_rope_index(
            iid, image_grid_thw=img_grid, video_grid_thw=vgrid,
            attention_mask=torch.ones_like(iid), mm_token_type_ids=mmtt)
        vpos = (iid[0] == self.video_token_id).nonzero().squeeze(-1)
        if n_slots < 1 or vpos.numel() % n_slots:
            raise RuntimeError(
                f"{vpos.numel()} video tokens do not tile into {n_slots} patches")
        per = vpos.numel() // n_slots
        slot_rows = [vpos[per * s: per * (s + 1)] for s in range(n_slots)]
        S1 = int(slot_rows[n_slots - 2][-1]) + 1 if n_slots >= 2 else 0
        if S1 and int((iid[0, :S1] == self.image_token_id).sum()):
            raise RuntimeError(
                "wrist image tokens fall inside the prefetched prefix; the "
                "prompt must place every current-frame camera AFTER the history "
                "video block")
        return {"iid": iid, "pos": pos, "mmtt": mmtt, "vgrid": vgrid,
                "img_grid": img_grid, "deltas": deltas, "per": per, "S1": S1,
                "n_slots": n_slots}

    def _layout_at_decision(self, prep, n_slots: int) -> dict:
        dev = self.dev
        iid = prep["input_ids"].to(dev)
        cached = self._layouts.get(n_slots)
        if cached is not None and torch.equal(cached["iid"], iid):
            return cached
        if cached is not None and not self._warned_layout:
            self._warned_layout = True
            print("[pipeline] WARNING: the pre-rendered layout does not match the "
                  "real prompt ids; rebuilding inline (correct, but the prefetch "
                  "is wasted). The layout is supposed to be a pure function of "
                  "(instruction, clip length).", flush=True)
        img_grid = prep["image_grid_thw"].to(dev) if "image_grid_thw" in prep else None
        layout = self._make_layout(
            iid, prep["mm_token_type_ids"].to(dev), prep["video_grid_thw"].to(dev),
            img_grid, n_slots)
        self._layouts[n_slots] = layout
        return layout

    def _encode_missing(self, pol, patch_pixels: dict) -> None:
        miss = [(k, px) for k, px in patch_pixels.items()
                if k not in pol._patch_feat_cache]
        if not miss:
            return
        gh, gw = pol._prompt_cache["grid_hw"]
        rows = torch.cat([px for _, px in miss], 0).to(self.dev)
        grid = torch.tensor([[1, gh, gw]] * len(miss), device=self.dev, dtype=torch.long)
        feats = self.qwen.get_video_features(rows, grid, return_dict=True).pooler_output
        for (k, _), f in zip(miss, feats):
            pol._patch_feat_cache[k] = f

    def _build_prefix(self, layout: dict, keys_head: list[tuple], feat_cache) -> None:
        iid, S1 = layout["iid"], layout["S1"]
        emb = self.embed(iid[:, :S1])
        vmask = iid[0, :S1] == self.video_token_id
        feats = torch.cat([feat_cache[k] for k in keys_head], dim=0)
        if int(vmask.sum()) != feats.shape[0]:
            raise RuntimeError(
                f"prefix video tokens {int(vmask.sum())} != cached features "
                f"{feats.shape[0]}")
        emb[0][vmask] = feats.to(emb.dtype)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=layout["pos"][:, :, :S1], use_cache=True)
        self.prefix_cache = out.past_key_values
        self.prefix_keys = list(keys_head)
        self.prefix_layout = layout

    def _prefetch(self, pol) -> None:
        self.prefix_cache = self.prefix_keys = self.prefix_layout = None
        nxt = pol.next_window(self.execute_horizon)
        if nxt is None:
            return
        clip_frames, keys_next = nxt
        n_slots_next = len(keys_next)
        if n_slots_next < 2:
            return
        layout = self._layouts.get(n_slots_next)
        if layout is None:
            prompt = pol._prompt_cache["key"][0]
            enc = pol._render_layout(prompt, clip_frames)
            img_grid = enc["image_grid_thw"] if "image_grid_thw" in enc else None
            layout = self._make_layout(
                enc["input_ids"], enc["mm_token_type_ids"], enc["video_grid_thw"],
                img_grid, n_slots_next)
            self._layouts[n_slots_next] = layout
        head = list(keys_next[:-1])
        self._encode_missing(
            pol, {k: pol._preprocess_patch_pixels(k)
                  for k in head if k not in pol._patch_feat_cache})
        self._build_prefix(layout, head, pol._patch_feat_cache)

    @torch.no_grad()
    def step(self, prep, state_norm, seed):
        dev = self.dev
        pol = prep["policy"]
        keys = prep["video_patch_keys"]
        n_slots = len(keys)

        self._encode_missing(pol, prep["video_patch_pixels"])
        layout = self._layout_at_decision(prep, n_slots)
        S1 = layout["S1"]

        torch.cuda.synchronize()
        t0 = time.time()
        hit = (self.prefix_cache is not None and self.prefix_layout is layout
               and self.prefix_keys == list(keys[:-1]))
        if not S1:
            self.n_nosplit += 1
        elif hit:
            self.n_hit += 1
        else:
            self.n_miss += 1
            self._build_prefix(layout, list(keys[:-1]), pol._patch_feat_cache)
        cache = self.prefix_cache if S1 else None
        self.prefix_cache = None

        iid, pos = layout["iid"], layout["pos"]
        emb = self.embed(iid[:, S1:])
        tail = iid[0, S1:]
        vmask = tail == self.video_token_id
        expected_v = layout["per"] if S1 else layout["per"] * n_slots
        if int(vmask.sum()) != expected_v:
            raise RuntimeError(
                f"the fed span holds {int(vmask.sum())} video tokens, expected "
                f"{expected_v}")
        tail_keys = keys[-1:] if S1 else list(keys)
        emb[0][vmask] = torch.cat(
            [pol._patch_feat_cache[k] for k in tail_keys], dim=0).to(emb.dtype)
        img_grid = layout["img_grid"]
        if img_grid is not None:
            ifeat = self._flat(self.qwen.get_image_features(
                prep["pixel_values"].to(dev), img_grid, return_dict=True).pooler_output)
            imask = tail == self.image_token_id
            if int(imask.sum()) != ifeat.shape[0]:
                raise RuntimeError("wrist image tokens and ViT features disagree")
            emb[0][imask] = ifeat.to(emb.dtype)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=pos[:, :, S1:], past_key_values=cache,
                      use_cache=True)
        result = self._decode_and_act(layout, out, out.past_key_values,
                                      state_norm, seed)
        torch.cuda.synchronize()
        t1 = time.time()

        self._prefetch(pol)
        torch.cuda.synchronize()
        t2 = time.time()

        self.t_crit.append(t1 - t0)
        self.t_bg.append(t2 - t1)
        return result

    def _decode_and_act(self, layout, prefill_out, cache, state_norm, seed):
        dev = self.dev
        iid = layout["iid"]
        L = iid.shape[1]
        max_new = self.max_subtask_tokens

        logits = self.lm_head(prefill_out.last_hidden_state[:, -1]).float()
        if self.suppress_token_ids:
            logits[:, self.suppress_token_ids] = float("-inf")
        sampled = [int(logits.argmax(-1)[0])]
        eos_seen = sampled[0] == self.im_end_id
        newline_fed = False
        runaway = False
        dec_hidden = []
        delta = layout["deltas"][:, 0].to(dev)
        fed = 0
        for step in range(max_new + 2):
            if step < len(sampled):
                tok_id = sampled[step]
            elif eos_seen and not newline_fed and self.newline_id is not None:
                tok_id = self.newline_id
                newline_fed = True
            else:
                break
            fed += 1
            p = (delta + L + fed - 1).view(1, 1, 1).expand(3, 1, 1)
            out = self.lm(input_ids=torch.tensor([[tok_id]], device=dev),
                          attention_mask=None, position_ids=p,
                          past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
            dec_hidden.append(h)
            step_logits = self.lm_head(h).float()
            if self.suppress_token_ids:
                step_logits[:, self.suppress_token_ids] = float("-inf")
            nxt = int(step_logits.argmax(-1)[0])
            if eos_seen or step != len(sampled) - 1:
                continue
            if len(sampled) < max_new:
                sampled.append(nxt)
                if nxt == self.im_end_id:
                    eos_seen = True
            else:
                sampled.append(self.im_end_id)
                eos_seen = True
                runaway = True

        if runaway:
            self.n_runaway += 1
            print(
                f"[pipeline] WARNING: sub-task decode hit the {max_new}-token "
                "budget without emitting <|im_end|>; forced a terminator so the "
                "DiT condition keeps its trained shape, but this decision is "
                "conditioned on a degenerate sub-task.", flush=True)

        gen_ids = torch.tensor(sampled, dtype=iid.dtype, device=dev)
        full = torch.cat([iid[0], gen_ids])
        if (self.newline_id is not None and gen_ids.numel() > 0
                and int(full[-1]) == self.im_end_id):
            full = torch.cat(
                [full, torch.tensor([self.newline_id], device=dev, dtype=full.dtype)])
        gen_len = full.shape[0] - L
        if gen_len > len(dec_hidden):
            raise RuntimeError("decode hidden capture shorter than generated span")

        fiid = full[None]
        fatt = torch.ones_like(fiid)
        fmmtt = torch.cat([layout["mmtt"][0],
                           torch.zeros(gen_len, dtype=layout["mmtt"].dtype,
                                       device=dev)])[None]
        span = subtask_span_mask(full, self.think_close_id, self.whitespace_ids)[None].to(dev)
        mm_full = {"image_grid_thw": layout["img_grid"],
                   "video_grid_thw": layout["vgrid"], "mm_token_type_ids": fmmtt}
        pos_full = self.vla._position_ids(fiid, fatt, mm_full)
        hid = torch.stack(dec_hidden, 1)
        hidden_full = hid.new_zeros(1, fiid.shape[1], hid.shape[-1])
        hidden_full[0, L: L + gen_len] = hid[0, :gen_len]
        cond_hidden, cond_token, cond_mask, cond_pos = self.vla._gather_condition(
            hidden_states=hidden_full, input_ids=fiid, span=span,
            position_ids=pos_full, attention_mask=fatt)
        torch.manual_seed(seed)
        actions = self.vla.action_head.sample(
            cond_hidden, condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask, condition_token_embeds=cond_token,
            num_steps=self.num_denoising_steps, temperature=self.eval_temperature,
            state=state_norm)
        subtask = self.processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True).strip()
        return subtask, actions


class PipelinedEvalGroup:

    def __init__(self, model, processor, unnormalize_action, normalize_state,
                 num_denoising_steps: int = 10, eval_temperature: float = 1.0,
                 max_subtask_tokens: int = 256, execute_horizon: int = 8,
                 device: torch.device | None = None):
        self.model = model
        self.processor = processor
        self.unnormalize = unnormalize_action
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.eval_temperature = float(eval_temperature)
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.execute_horizon = int(execute_horizon)
        self.device = device or next(model.parameters()).device
        cfg = model.config
        self.use_proprio = bool(cfg.use_proprio)
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self._decisions = 0
        self._crit: deque[float] = deque(maxlen=4000)
        self._bg: deque[float] = deque(maxlen=4000)
        self._hit = self._miss = self._nosplit = 0

    def _runner_for(self, pol: PipelinedBufferPolicy) -> PipelinedKVRunner:
        if pol._stream_runner is None:
            pol._stream_runner = PipelinedKVRunner(
                self.model, self.processor, self.normalize_state,
                self.num_denoising_steps, self.max_subtask_tokens,
                self.eval_temperature, self.execute_horizon)
        return pol._stream_runner

    def _report(self) -> None:
        if not self._crit:
            return
        budget = self.execute_horizon / 20.0
        split = self._hit + self._miss
        print(f"[pipeline] n={len(self._crit)} critical={np.median(self._crit):.3f}s "
              f"background={np.median(self._bg):.3f}s (medians) "
              f"prefetch_hit={self._hit}/{split} nosplit={self._nosplit} "
              f"(deployment latency = critical; {budget:.2f}s budget for background)",
              flush=True)

    @torch.no_grad()
    def generate_batch(self, processed_list, state_norm_list):
        results = []
        for prep, state in zip(processed_list, state_norm_list):
            pol: PipelinedBufferPolicy = prep["policy"]
            runner = self._runner_for(pol)
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            before = (runner.n_hit, runner.n_miss, runner.n_nosplit)
            subtask, actions = runner.step(prep, state, seed)
            self._crit.append(runner.t_crit[-1])
            self._bg.append(runner.t_bg[-1])
            self._hit += runner.n_hit - before[0]
            self._miss += runner.n_miss - before[1]
            self._nosplit += runner.n_nosplit - before[2]
            norm = actions[0, :, : self.action_dim].float().cpu()
            unnorm = self.unnormalize.unnormalize(norm).numpy().astype(np.float32)
            results.append((unnorm, subtask))
        self._decisions += 1
        if self._decisions % 25 == 0:
            self._report()
        return results
