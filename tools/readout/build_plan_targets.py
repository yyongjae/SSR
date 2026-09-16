"""Extract command / ego status / trajectory targets for every token of a split.

CPU only.  One pass over the nuPlan logs; readout training never opens them.

    python tools/readout/build_plan_targets.py --filter navtrain --split trainval \
        --out /data3/kyungmin/readout/plan_targets_navtrain.npz
    python tools/readout/build_plan_targets.py --filter navtest --split test \
        --out /data3/kyungmin/readout/plan_targets_navtest.npz
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import _env  # noqa: F401
from _env import scene_filter, split_dirs


def _work(job):
    filter_name, split, logs, tokens, num_poses = job
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.readout.plan_targets import plan_arrays_from_scene

    logs_dir, blobs_dir = split_dirs(split)
    sf = scene_filter(filter_name, log_names=logs, tokens=tokens)
    loader = SceneLoader(logs_dir, blobs_dir, sf)
    out = []
    for log, toks in loader.get_tokens_list_per_log().items():
        for tok in toks:
            scene = loader.get_scene_from_token(tok)
            out.append((tok, log, plan_arrays_from_scene(scene, num_poses)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="navtrain")
    ap.add_argument("--split", default="trainval", help="navsim_logs/<split>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--logs-per-job", type=int, default=20)
    ap.add_argument("--max-logs", type=int, default=0, help="smoke test: first N logs only")
    ap.add_argument("--num-poses", type=int, default=8)
    args = ap.parse_args()

    from navsim.agents.para_ssr.readout.plan_targets import PlanTargetStore

    sf = scene_filter(args.filter)
    logs_dir, _ = split_dirs(args.split)
    logs = sf.log_names or sorted(p.stem for p in logs_dir.glob("*.pkl"))
    if args.max_logs:
        logs = logs[: args.max_logs]
    jobs = [
        (args.filter, args.split, logs[i : i + args.logs_per_job], sf.tokens, args.num_poses)
        for i in range(0, len(logs), args.logs_per_job)
    ]
    rows, tokens, log_of = [], [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, part in enumerate(pool.imap_unordered(_work, jobs), 1):
            for tok, log, arr in part:
                tokens.append(tok)
                log_of.append(log)
                rows.append(arr)
            print(f"[{n}/{len(jobs)}] {len(tokens)} tokens", flush=True)
    order = sorted(range(len(tokens)), key=lambda i: tokens[i])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    PlanTargetStore.save(
        Path(args.out), [tokens[i] for i in order], [log_of[i] for i in order], [rows[i] for i in order]
    )
    print(f"wrote {len(tokens)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
