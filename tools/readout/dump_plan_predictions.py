"""Dump a PARA-SSR checkpoint's planned trajectories for every token of a split.

The PDMS evaluation scores trajectories without keeping them; the failure
analysis (data/analysis/failure_attribution.py) needs them next to the det/map
predictions that ``run_aux_evaluation`` already stores per token.

    CUDA_VISIBLE_DEVICES=3 python tools/readout/dump_plan_predictions.py \
        --ckpt <last.ckpt> --filter navtest --split test --out data/analysis/plan_control_ft.npz \
        agent.config.use_task_interaction=true agent.config.use_det_motion_head=true \
        agent.config.use_map_head=true

Saved arrays (token order sorted): ``tokens``, ``trajectory`` [N, T, 3] (the
commanded branch, NAVSIM ego frame: x forward, y left, rear axle, as scored),
``ego_fut_preds`` [N, num_cmd, T, 3] per-step offsets of every branch, ``command`` [N, num_cmd].
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _env  # noqa: F401
from _env import scene_filter, split_dirs
from cache_student_bev import FeatureData, build_agent, collate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--filter", default="navtest")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--v2-extras", action="store_true",
                    help="also save the anchor planner's candidates and scores (top-k, all offsets, rewards)")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(args.ckpt, overrides).to(device)
    logs_dir, blobs_dir = split_dirs(args.split)
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter(args.filter), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.limit:
        tokens = tokens[: args.limit]
    dl = DataLoader(FeatureData(tokens, loader, agent.get_feature_builders()), batch_size=args.batch_size,
                    num_workers=args.workers, collate_fn=collate)
    names, traj, preds, cmds, missing = [], [], [], [], []
    extras = {k: [] for k in ("plan_topk_index", "plan_topk_trajectory", "plan_final_rewards", "sim_rewards",
                              "im_rewards", "trajectory_offset")} if args.v2_extras else {}
    t0 = time.time()
    with torch.no_grad():
        for toks, feats, failed in dl:
            missing.extend(t for t, _ in failed)
            if feats is None:
                continue
            feats = {k: v.to(device) for k, v in feats.items()}
            out = agent.para_ssr_model(feats, run_aux=False)
            names.extend(toks)
            traj.append(out["trajectory"].float().cpu().numpy())
            preds.append(out["ego_fut_preds"].float().cpu().numpy())
            cmds.append(feats["command"].float().cpu().numpy())
            for k in extras:
                v = out[k].cpu().numpy()
                extras[k].append(v if v.dtype.kind in "iu" else v.astype(np.float16))
            if len(names) % 2000 < args.batch_size:
                print(f"{len(names)}/{len(tokens)}  {(time.time() - t0) / len(names):.3f} s/frame", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=np.asarray(names), trajectory=np.concatenate(traj),
                        ego_fut_preds=np.concatenate(preds), command=np.concatenate(cmds),
                        missing=np.asarray(missing), checkpoint=np.asarray(args.ckpt),
                        **{k: np.concatenate(v) for k, v in extras.items()})
    print(f"saved {len(names)} trajectories ({len(missing)} missing) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
