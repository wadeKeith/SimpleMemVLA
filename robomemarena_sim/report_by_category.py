
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

CATEGORIES: dict[str, tuple[int, ...]] = {
    "Transferring": (18, 19, 25, 26),
    "Occlusion": (4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24),
    "Counting": (6, 7, 8, 9, 10, 15, 16),
    "Sequence": (1, 2, 3, 22),
}

PUBLISHED = [
    ("pi_0.5 (Intelligence et al., 2025)", (20.0, 42.8, 12.7, 17.2, 14.3, 50.9, 60.0, 71.6, 21.5, 38.7)),
    ("HiF-VLA (Lin et al., 2025b)",        (17.5, 38.9, 12.7, 27.1,  8.6, 45.9, 42.5, 70.2, 16.9, 39.8)),
    ("MemoryVLA (Shi et al., 2026a)",      (15.0, 37.2,  7.3, 13.1, 14.3, 55.1, 37.5, 65.2, 15.0, 35.3)),
    ("MemER (Sridhar et al., 2026)",       (20.0, 36.1, 16.4, 33.2, 27.1, 65.1, 65.0, 79.1, 27.3, 49.1)),
    ("PrediMem (Lei et al., 2026a)",       (22.5, 45.2, 27.3, 38.4, 45.7, 69.3, 72.5, 89.5, 38.5, 55.2)),
    ("Ground truth (oracle)",              (32.5, 54.8, 33.6, 49.8, 51.4, 75.6, 85.0, 92.3, 46.1, 64.8)),
]


def load_tasks(out_root: Path) -> dict[int, list[dict]]:
    shards = sorted((out_root / "shards").glob("*.json"))
    if not shards:
        raise FileNotFoundError(f"no shard JSONs under {out_root}/shards")
    by_task: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    for path in shards:
        payload = json.loads(path.read_text())
        tid = int(payload["task_id"])
        for ep in payload.get("episodes", []):
            by_task[tid][int(ep["ep"])] = ep
    return {t: [eps[k] for k in sorted(eps)] for t, eps in by_task.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--label", default="SimpleMemVLA (Ours)")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    by_task = load_tasks(args.out_root)
    assigned = {t for ids in CATEGORIES.values() for t in ids}
    if assigned != set(range(1, 27)):
        raise ValueError(f"category map does not cover tasks 1..26: {sorted(assigned)}")

    per_task = {
        t: (
            sum(float(e.get("TSR", 0.0)) for e in eps) / len(eps),
            sum(float(e.get("CSR", 0.0)) for e in eps) / len(eps),
            len(eps),
        )
        for t, eps in by_task.items()
    }
    missing = [t for t in range(1, 27) if t not in per_task]
    short = {t: n for t, (_, _, n) in per_task.items() if n != 51}

    cells: list[float] = []
    detail = []
    for name, ids in CATEGORIES.items():
        have = [t for t in ids if t in per_task]
        if not have:
            cells += [float("nan"), float("nan")]
            detail.append((name, ids, have, float("nan"), float("nan")))
            continue
        tsr = sum(per_task[t][0] for t in have) / len(have)
        csr = sum(per_task[t][1] for t in have) / len(have)
        cells += [tsr, csr]
        detail.append((name, ids, have, tsr, csr))

    have_all = [t for t in range(1, 27) if t in per_task]
    avg_tsr = sum(per_task[t][0] for t in have_all) / len(have_all)
    avg_csr = sum(per_task[t][1] for t in have_all) / len(have_all)
    row = cells + [avg_tsr, avg_csr]

    head = f"{'Method':<36}" + "".join(f"{c:>16}" for c in
                                       ["Transferring", "Occlusion", "Counting", "Sequence", "Average"])
    sub = f"{'':<36}" + "".join(f"{'TSR':>8}{'CSR':>8}" for _ in range(5))
    print(head); print(sub); print("-" * len(sub))
    if not args.no_baselines:
        for name, vals in PUBLISHED:
            print(f"{name:<36}" + "".join(f"{v:>8.1f}" for v in vals))
        print("-" * len(sub))
    print(f"{args.label:<36}" + "".join(f"{v:>8.1f}" for v in row))
    print()
    print("per-category task membership and per-task detail:")
    for name, ids, have, tsr, csr in detail:
        print(f"  {name:<13} n_tasks={len(have)}/{len(ids)}  TSR={tsr:5.1f}  CSR={csr:5.1f}")
        for t in ids:
            if t in per_task:
                a, b, n = per_task[t]
                print(f"      task{t:<3d} TSR={a:5.1f}  CSR={b:5.1f}  (n={n})")
            else:
                print(f"      task{t:<3d} -- no episodes --")
    if missing:
        print(f"\nWARNING: tasks with no episodes at all: {missing}")
    if short:
        print(f"WARNING: tasks with n != 51: {short}")
    if missing or short:
        print("Numbers above are PARTIAL -- do not report them as a full run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
