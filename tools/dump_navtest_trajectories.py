#!/usr/bin/env python3
"""Dump one PARA-SSR arm's navtest trajectories for scoring outside this repo.

NAVSIM v2 scores EPDMS, but its evaluation instantiates the agent inside CPU
worker processes -- unusable for a model whose LiDAR encoder is CUDA-only
(spconv).  This tool runs the exact inference the v1 PDMS evaluation runs
(same checkpoint, same feature builders, same archived training config), on
one GPU with a batched dataloader, and writes

    {"trajectories": {token: float32 [num_poses, 3]}, "meta": {...}}

which navsim_v2's run_pdm_score_from_trajectories.py scores with the unmodified
v2 simulator, scorer and two-frame extended-comfort aggregation.  Only the
trajectory source differs from v2's own one-stage runner.

    python tools/dump_navtest_trajectories.py --arm ssr --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARMS = {
    "ssr": "para_ssr_front3_lidar_30ep",
    "nodet": "para_ssr_map_plan",
    "nomap": "para_ssr_det_motion_plan",
    "plan_only": "para_ssr_plan_only",
}


class _TokenFeatures(Dataset):
    def __init__(self, loader, builders, tokens):
        self._loader, self._builders, self._tokens = loader, builders, tokens

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        token = self._tokens[index]
        agent_input = self._loader.get_agent_input_from_token(token)
        features = {}
        for builder in self._builders:
            features.update(builder.compute_features(agent_input))
        return token, features


def _collate(batch):
    tokens = [token for token, _ in batch]
    keys = batch[0][1].keys()
    return tokens, {k: torch.stack([features[k] for _, features in batch]) for k in keys}


def _worker_init(_):
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: newest version's last.ckpt")
    parser.add_argument(
        "--download", type=Path, default=os.environ.get("NAVSIM_DOWNLOAD") or None,
        help="unpacked NAVSIM download holding test_navsim_logs/test and test_sensor_blobs/test "
             "(default: $NAVSIM_DOWNLOAD, else data/dataset/{navsim_logs,sensor_blobs}/test)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO_ROOT / "data/dataset/maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO_ROOT / "data/dataset"))

    if args.download is not None:
        log_path = Path(args.download) / "test_navsim_logs/test"
        blob_path = Path(args.download) / "test_sensor_blobs/test"
    else:
        data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
        log_path, blob_path = data_root / "navsim_logs/test", data_root / "sensor_blobs/test"
    for path in (log_path, blob_path):
        if not path.is_dir():
            parser.error(f"navtest data not found: {path} (set --download or NAVSIM_DOWNLOAD)")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader

    experiment = ARMS[args.arm]
    run_dir = REPO_ROOT / "work_dirs" / experiment
    checkpoint = args.checkpoint or sorted(
        run_dir.glob("lightning_logs/version_*/checkpoints/last.ckpt"),
        key=lambda p: int(p.parts[-3].split("_")[1]),
    )[-1]
    training_config = OmegaConf.load(run_dir / "code/hydra/config.yaml")
    agent_cfg = OmegaConf.create(OmegaConf.to_container(training_config.agent, resolve=False))
    OmegaConf.set_struct(agent_cfg, False)
    agent_cfg.checkpoint_path = str(checkpoint)
    agent_cfg.resume_from_checkpoint = False
    agent_cfg.config.backbone_pretrained = False
    agent_cfg.config.test_aux_heads = False       # the trajectory never reads the aux heads

    device = torch.device(args.device)
    agent = instantiate(agent_cfg)
    agent.initialize()                            # strict load
    agent = agent.to(device).eval()
    ckpt_epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"])

    scene_filter = instantiate(OmegaConf.load(
        REPO_ROOT / "navsim/planning/script/config/common/scene_filter/navtest.yaml"))
    loader = SceneLoader(
        sensor_blobs_path=blob_path,
        data_path=log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens = sorted(loader.tokens)
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    print(f"[{args.arm}] {experiment} epoch {ckpt_epoch} | {len(tokens)} tokens | {checkpoint}", flush=True)

    data = DataLoader(
        _TokenFeatures(loader, agent.get_feature_builders(), tokens),
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=_collate,
        worker_init_fn=_worker_init, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    trajectories, start = {}, time.time()
    with torch.no_grad():
        for batch_tokens, features in data:
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            poses = agent.para_ssr_model(features, run_aux=False)["trajectory"].float().cpu().numpy()
            if not np.isfinite(poses).all():
                raise RuntimeError(f"non-finite trajectory in batch starting at {batch_tokens[0]}")
            for token, pose in zip(batch_tokens, poses):
                trajectories[token] = pose.astype(np.float32)
            if len(trajectories) % 2000 < args.batch_size:
                rate = len(trajectories) / (time.time() - start)
                print(f"  {len(trajectories)}/{len(tokens)}  {rate:.1f} tok/s", flush=True)

    sampling = agent._trajectory_sampling
    meta = {
        "arm": args.arm,
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt_epoch,
        "use_det_motion_head": bool(agent_cfg.config.use_det_motion_head),
        "use_map_head": bool(agent_cfg.config.use_map_head),
        "num_poses": int(sampling.num_poses),
        "interval_length": float(sampling.interval_length),
        "time_horizon": float(sampling.time_horizon),
        "frame": "NAVSIM ego (x forward, y left, heading)",
        "num_tokens": len(trajectories),
        "token_sha256": hashlib.sha256("".join(f"{t}\n" for t in sorted(trajectories)).encode()).hexdigest(),
    }
    out = args.out or REPO_ROOT / "work_dirs/eval" / f"{experiment}_navtest_trajectories.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"trajectories": trajectories, "meta": meta}, f)
    print(f"[{args.arm}] wrote {len(trajectories)} trajectories to {out} in {time.time() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
