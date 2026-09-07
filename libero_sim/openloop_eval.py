
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libero_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from transformers import AutoProcessor

from libero_sim.policy import LiberoPolicy, get_vla
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from simplememvla.benchmarks.libero import NUM_ACTIONS_CHUNK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--root", default="./data/datasets/yinchenghust/libero_lerobot")
    parser.add_argument("--repo_id", default="yinchenghust/libero_lerobot")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None,
        help="Explicit episode indices to eval (overrides --num_episodes).",
    )
    parser.add_argument("--compute_dtype", default="bfloat16")
    parser.add_argument("--num_denoising_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_reasoning_tokens", type=int, default=256)
    parser.add_argument(
        "--num_open_loop_steps", type=int, default=8,
        help="Re-predict every N frames (receding horizon). Defaults to 8 to match "
             "the closed-loop eval's --execute_horizon, so this open-loop score is "
             "a faithful quality proxy for closed loop (same re-decision cadence).",
    )
    parser.add_argument("--print_samples", type=int, default=3,
                        help="Print the first N generated-vs-GT reasoning pairs per episode.")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument(
        "--attn_implementation", default="flash_attention_2",
        help="Backbone attention backend (default flash_attention_2, matching training; "
             "use sdpa on a host without flash-attn).",
    )
    return parser.parse_args()


def _to_uint8_hwc(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        if t.dtype.is_floating_point:
            t = t.clamp(0, 1).mul(255).round().to(torch.uint8)
        else:
            t = t.to(torch.uint8)
        return t.numpy()
    return np.asarray(img)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def main() -> None:
    args = parse_args()
    vla, unnormalize_action, normalize_state = get_vla(
        args.pretrained_checkpoint, args.compute_dtype,
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(
        args.pretrained_checkpoint,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=8192,
    )
    policy = LiberoPolicy(
        vla=vla,
        processor=processor,
        unnormalize_action=unnormalize_action,
        normalize_state=normalize_state,
        num_denoising_steps=args.num_denoising_steps,
        temperature=args.temperature,
        max_reasoning_tokens=args.max_reasoning_tokens,
    )

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = meta.fps
    action_delta = {"action": [i / fps for i in range(NUM_ACTIONS_CHUNK)]}

    if args.episodes is not None:
        episodes = args.episodes
    else:
        n = min(args.num_episodes, meta.total_episodes)
        episodes = list(range(n))
    print(f"Evaluating {len(episodes)} episode(s) on {args.repo_id}: {episodes}")

    tot_l1, tot_steps, tot_hit, tot_pred = 0.0, 0, 0, 0
    for ep in episodes:
        ds = LeRobotDataset(
            args.repo_id, root=args.root, episodes=[ep],
            delta_timestamps=action_delta, video_backend=args.video_backend,
        )
        policy.reset()
        ep_l1, ep_steps, ep_hit, ep_pred = 0.0, 0, 0, 0
        printed = 0
        for i in range(len(ds)):
            item = ds[i]
            policy.observe({k: _to_uint8_hwc(item[k]) for k in policy.image_keys})
            if i % args.num_open_loop_steps != 0:
                continue
            state = np.asarray(item["observation.state"], dtype=np.float32)
            pred_actions, gen_subtask = policy.predict(item["task"], state=state)
            gt = item["action"].numpy().astype(np.float32)
            valid = ~item["action_is_pad"].numpy().astype(bool)
            pred = np.stack(pred_actions)[: gt.shape[0]].astype(np.float32)
            if valid.any():
                ep_l1 += float(np.abs(pred[valid] - gt[valid]).sum())
                ep_steps += int(valid.sum()) * gt.shape[1]
            gt_subtask = str(item["subtask"]).strip()
            ep_hit += int(_norm(gen_subtask) == _norm(gt_subtask))
            ep_pred += 1
            if printed < args.print_samples:
                printed += 1
                print(f"    [ep {ep} frame {i}]")
                print(f"      GT : {gt_subtask[:200]}")
                print(f"      GEN: {gen_subtask[:200]}")
        l1 = ep_l1 / max(ep_steps, 1)
        acc = ep_hit / max(ep_pred, 1)
        print(f"  ep {ep:>3}: action_L1={l1:.4f}  subtask_exact={acc:.3f} "
              f"({ep_hit}/{ep_pred})  predictions={ep_pred}")
        tot_l1 += ep_l1
        tot_steps += ep_steps
        tot_hit += ep_hit
        tot_pred += ep_pred

    print("=" * 60)
    print(f"Overall action_L1     : {tot_l1 / max(tot_steps, 1):.4f} "
          "(raw OSC_POSE delta action space)")
    print(f"Overall subtask_exact : {tot_hit / max(tot_pred, 1):.3f} "
          f"({tot_hit}/{tot_pred})")


if __name__ == "__main__":
    main()
