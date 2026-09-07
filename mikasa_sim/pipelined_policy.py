
from __future__ import annotations

import copy
import statistics
import time

import torch

from mikasa_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from simplememvla.data.collator import subtask_span_mask
from mikasa_sim.policy import MikasaPolicy


def _flat(feat) -> torch.Tensor:
    if isinstance(feat, (list, tuple)):
        return torch.cat(list(feat), dim=0)
    if feat.dim() == 3:
        return feat.flatten(0, 1)
    return feat


class PipelinedKVRunner:

    def __init__(
        self,
        vla,
        processor,
        num_denoising_steps: int,
        temperature: float,
        max_subtask_tokens: int,
        execute_horizon: int,
        think_close_id: int | None,
        whitespace_ids: tuple[int, ...],
        newline_id: int | None,
        im_end_id: int,
    ):
        self.vla = vla
        self.processor = processor
        self.num_denoising_steps = int(num_denoising_steps)
        self.temperature = float(temperature)
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.execute_horizon = int(execute_horizon)
        self.think_close_id = think_close_id
        self.whitespace_ids = whitespace_ids
        self.newline_id = newline_id
        self.im_end_id = im_end_id

        self.qwen = vla.backbone.model
        self.lm = self.qwen.language_model
        self.lm_head = vla.backbone.get_output_embeddings()
        self.embed = self.qwen.get_input_embeddings()
        self.device = next(vla.parameters()).device
        self._t_critical: list[float] = []
        self._t_background: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.prefix_cache = None
        self.prefix_keys: list[tuple] = []
        self.prefix_end = 0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _encode_missing_patches(self, prep) -> None:
        pol: MikasaPolicy = prep["policy"]
        missing = list(prep["video_patch_pixels"].items())
        if not missing:
            return
        gh, gw = pol.grid_hw
        rows = torch.cat([px for _, px in missing], dim=0).to(self.device)
        grid = torch.tensor([[1, gh, gw]] * len(missing), device=self.device,
                            dtype=torch.long)
        feats = self.qwen.get_video_features(rows, grid, return_dict=True).pooler_output
        if len(feats) != len(missing):
            raise RuntimeError(
                f"video ViT returned {len(feats)} feature blocks for {len(missing)} patches"
            )
        for (key, _), feat in zip(missing, feats):
            pol._patch_feat_cache[key] = feat

    def _slot_end(self, layout: dict, n_slots: int) -> int:
        if n_slots <= 0:
            return 0
        return int(layout["slot_rows"][n_slots - 1][-1]) + 1

    def _video_embeds(self, layout: dict, keys, start: int, stop: int,
                      feat_cache: dict, first_slot: int) -> torch.Tensor:
        ids = layout["input_ids"]
        emb = self.embed(ids[:, start:stop])
        mask = ids[0, start:stop] == self.qwen.config.video_token_id
        n_slots = int(mask.sum()) // layout["tokens_per_slot"]
        if int(mask.sum()) % layout["tokens_per_slot"]:
            raise RuntimeError("fed span does not hold a whole number of temporal patches")
        wanted = list(keys[first_slot: first_slot + n_slots])
        if wanted:
            emb[0][mask] = torch.cat([feat_cache[k] for k in wanted], dim=0).to(emb.dtype)
        return emb

    def _build_prefix(self, layout: dict, keys, n_slots: int, feat_cache: dict):
        stop = self._slot_end(layout, n_slots)
        emb = self._video_embeds(layout, keys, 0, stop, feat_cache, 0)
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=layout["position_ids"][:, :, :stop], use_cache=True)
        return out.past_key_values

    @torch.no_grad()
    def step(self, prep: dict, state: torch.Tensor | None) -> tuple[torch.Tensor, str]:
        pol: MikasaPolicy = prep["policy"]
        layout = prep["layout"]
        keys = prep["video_patch_keys"]
        n_slots = len(keys)

        self._sync()
        t0 = time.perf_counter()
        self._encode_missing_patches(prep)

        reuse = (
            self.prefix_cache is not None
            and len(self.prefix_keys) <= n_slots
            and list(keys[: len(self.prefix_keys)]) == self.prefix_keys
        )
        if reuse:
            done = len(self.prefix_keys)
            cache = self.prefix_cache
            if self._slot_end(layout, done) != self.prefix_end:
                raise RuntimeError(
                    "carried prefix covers a different token count in this window's "
                    f"layout ({self._slot_end(layout, done)} vs {self.prefix_end}); "
                    "per-slot blocks must be layout-invariant"
                )
        else:
            done = 0
            cache = None

        video_end = self._slot_end(layout, n_slots)
        start = self._slot_end(layout, done)
        if start < video_end:
            emb = self._video_embeds(layout, keys, start, video_end,
                                     pol._patch_feat_cache, done)
            out = self.lm(inputs_embeds=emb, attention_mask=None,
                          position_ids=layout["position_ids"][:, :, start:video_end],
                          past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        self.prefix_cache = copy.deepcopy(cache)
        self.prefix_keys = list(keys)
        self.prefix_end = video_end

        ids = layout["input_ids"]
        emb = self.embed(ids[:, video_end:])
        image_mask = ids[0, video_end:] == self.qwen.config.image_token_id
        if "pixel_values" in prep:
            ifeat = _flat(self.qwen.get_image_features(
                prep["pixel_values"], prep["image_grid_thw"], return_dict=True
            ).pooler_output)
            if int(image_mask.sum()) != ifeat.shape[0]:
                raise RuntimeError("wrist image tokens and ViT features disagree")
            emb[0][image_mask] = ifeat.to(emb.dtype)
        elif bool(image_mask.any()):
            raise RuntimeError("prompt has image tokens but no wrist pixels were prepared")
        out = self.lm(inputs_embeds=emb, attention_mask=None,
                      position_ids=layout["position_ids"][:, :, video_end:],
                      past_key_values=cache, use_cache=True)

        actions, subtask = self._decode_and_act(layout, out, out.past_key_values, state)
        self._sync()
        t1 = time.perf_counter()

        self._prefetch_next_prefix(pol, prep)
        self._sync()
        t2 = time.perf_counter()

        self._t_critical.append(t1 - t0)
        self._t_background.append(t2 - t1)
        if len(self._t_critical) % 25 == 0:
            print(f"[pipeline] n={len(self._t_critical)} "
                  f"critical={statistics.mean(self._t_critical) * 1e3:.0f} ms "
                  f"background={statistics.mean(self._t_background) * 1e3:.0f} ms "
                  f"(deployment latency = critical; the background phase hides "
                  f"inside the {self.execute_horizon / 20.0:.2f} s execution window)",
                  flush=True)
        return actions, subtask

    def _prefetch_next_prefix(self, pol: MikasaPolicy, prep: dict) -> None:
        try:
            next_keys = pol.future_patch_keys(self.execute_horizon)
        except RuntimeError:
            return
        if list(next_keys[: len(self.prefix_keys)]) == self.prefix_keys:
            return
        cached = pol._patch_feat_cache
        shared = 0
        while shared < len(next_keys) and next_keys[shared] in cached:
            shared += 1
        if shared == 0:
            self.prefix_cache, self.prefix_keys, self.prefix_end = None, [], 0
            return
        layout = pol._layout(prep["layout"]["prompt"],
                             len(next_keys) // len(pol.history_image_keys))
        self.prefix_cache = self._build_prefix(layout, next_keys, shared, cached)
        self.prefix_keys = list(next_keys[:shared])
        self.prefix_end = self._slot_end(layout, shared)

    def _decode_and_act(self, layout: dict, prefill_out, cache, state):
        ids = layout["input_ids"]
        L = ids.shape[1]
        newline_id = self.newline_id
        im_end_id = self.im_end_id

        base = int(layout["rope_deltas"].reshape(-1)[0]) + L
        fed_ids: list[int] = []
        hiddens: list[torch.Tensor] = []
        token = int(self.lm_head(prefill_out.last_hidden_state[:, -1]).float().argmax(-1)[0])
        forced = False
        while True:
            pos = torch.full((3, 1, 1), base + len(fed_ids), dtype=torch.long,
                             device=self.device)
            out = self.lm(input_ids=torch.tensor([[token]], device=self.device),
                          attention_mask=None, position_ids=pos,
                          past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            hidden = out.last_hidden_state[:, -1]
            hiddens.append(hidden)
            fed_ids.append(token)
            if token == im_end_id:
                break
            if len(fed_ids) >= self.max_subtask_tokens:
                if not forced:
                    print(f"[policy] WARNING: sub-task decode hit the "
                          f"{self.max_subtask_tokens}-token budget without "
                          "<|im_end|>; forcing a terminator (degenerate sub-task).",
                          flush=True)
                    forced = True
                token = im_end_id
                continue
            token = int(self.lm_head(hidden).float().argmax(-1)[0])

        subtask = self.processor.tokenizer.decode(
            fed_ids, skip_special_tokens=True).strip()

        if newline_id is not None:
            pos = torch.full((3, 1, 1), base + len(fed_ids), dtype=torch.long,
                             device=self.device)
            out = self.lm(input_ids=torch.tensor([[newline_id]], device=self.device),
                          attention_mask=None, position_ids=pos,
                          past_key_values=cache, use_cache=True)
            hiddens.append(out.last_hidden_state[:, -1])
            fed_ids.append(newline_id)

        gen = torch.tensor(fed_ids, dtype=ids.dtype, device=self.device)
        full_ids = torch.cat([ids[0], gen])[None]
        attn = torch.ones_like(full_ids)
        mm_token_type_ids = torch.cat([
            layout["mm_token_type_ids"][0],
            torch.zeros(len(fed_ids), dtype=layout["mm_token_type_ids"].dtype,
                        device=self.device),
        ])[None]
        span = subtask_span_mask(full_ids[0], self.think_close_id,
                                 self.whitespace_ids)[None]
        if bool(span[0, :L].any()):
            raise RuntimeError(
                "sub-task condition span reaches into the prompt; the pipelined path "
                "only holds hidden states for the generated tokens"
            )
        position_ids = self.vla._position_ids(full_ids, attn, {
            "image_grid_thw": layout["image_grid_thw"],
            "video_grid_thw": layout["video_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
        })
        stacked = torch.stack(hiddens, dim=1)
        hidden_full = stacked.new_zeros(1, full_ids.shape[1], stacked.shape[-1])
        hidden_full[0, L:] = stacked[0]
        cond_hidden, cond_token, cond_mask, cond_pos = self.vla._gather_condition(
            hidden_states=hidden_full, input_ids=full_ids, span=span,
            position_ids=position_ids, attention_mask=attn)
        actions = self.vla.action_head.sample(
            cond_hidden, condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask, condition_token_embeds=cond_token,
            num_steps=self.num_denoising_steps, temperature=self.temperature,
            state=state)
        return actions, subtask


class PipelinedMikasaPolicy(MikasaPolicy):

    def __init__(self, *args, execute_horizon: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = PipelinedKVRunner(
            vla=self.vla,
            processor=self.processor,
            num_denoising_steps=self.num_denoising_steps,
            temperature=self.temperature,
            max_subtask_tokens=self.max_subtask_tokens,
            execute_horizon=execute_horizon,
            think_close_id=self.think_close_id,
            whitespace_ids=self.whitespace_ids,
            newline_id=self.newline_id,
            im_end_id=self.im_end_id,
        )

    def reset(self) -> None:
        super().reset()
        self.runner.reset()

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        state=None,
        observe=None,
    ) -> tuple[list, str]:
        if observe is not None:
            self.observe(observe)
        state_tensor = self._state_tensor(state)
        prep = self._prepare_inputs(prompt)
        normalized_actions, subtask = self.runner.step(prep, state_tensor)
        return self.unnormalize_chunk(normalized_actions), subtask
