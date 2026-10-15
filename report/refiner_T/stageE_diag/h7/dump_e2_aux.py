#!/usr/bin/env python3
"""H7: dump E2's own detection / map head outputs on navtest in the aux-record format (det_pred_*, map_pred_*),
so they can be matched against the same V3 GT as E0 (work_dirs/eval/para_ssr_interaction_final_aux/records).

Model build = dump_e2_navtest.py (archived E2 eval hydra config, strict load of last.ckpt, eval, fp32); forward =
agent._loss.apply_aux_scales + para_ssr_model(features, run_aux=True) (the aux-eval call, no autocast); decode =
navsim.evaluate.aux_metrics.decode_{detection,map}_predictions (top-100, as run_aux_evaluation.py).  GT is NOT
recomputed here (identical V3 GT is copied from the E0 aux record of the same token).  tau0 of this forward is stored
and compared with e2_navtest_dump.npz (sanity).
Output: <out>/records/<token>.npz, <out>/tau0.npz, <out>/meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from dump_navtest_trajectories import _TokenFeatures, _collate, _worker_init  # noqa: E402

RUN = REPO / "work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0"
EVAL_CFG = REPO / "work_dirs/eval/stageE_E2/code/hydra/config.yaml"
E0_REC = REPO / "work_dirs/eval/para_ssr_interaction_final_aux/records"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=RUN / "lightning_logs/version_0/checkpoints/last.ckpt")
    ap.add_argument("--download", type=Path, default=Path("/home/external-user/navsim/download"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--priority", default="", help="json token list processed first")
    ap.add_argument("--out", type=Path, default=Path("/home/external-user/ssd/yongjae_refiner/stageE_diag/h7/e2_aux"))
    a = ap.parse_args()
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))
    torch.set_num_threads(1)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.aux_metrics import decode_detection_predictions, decode_map_predictions

    cfg = OmegaConf.load(EVAL_CFG)
    agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=False))
    OmegaConf.set_struct(agent_cfg, False)
    agent_cfg.checkpoint_path = str(a.checkpoint)
    dev = torch.device(a.device)
    agent = instantiate(agent_cfg).to(dev)
    agent.initialize()
    agent.is_eval = True
    agent.eval()
    model = agent.para_ssr_model
    map_pc_range = tuple(float(v) for v in agent.config.map_pc_range)

    scene_filter = instantiate(cfg.scene_filter)
    loader = SceneLoader(sensor_blobs_path=a.download / "test_sensor_blobs/test",
                         data_path=a.download / "test_navsim_logs/test", scene_filter=scene_filter,
                         sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if a.max_tokens:
        tokens = tokens[: a.max_tokens]
    rec_dir = a.out / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    if a.priority:  # priority tokens first, then the rest in a fixed random order (so a partial run is a random sample)
        pri = [t for t in json.load(open(a.priority)) if t in set(tokens)]
        rest = sorted(set(tokens) - set(pri))
        rest = [rest[i] for i in np.random.default_rng(0).permutation(len(rest))]
        tokens = pri + rest
    todo = [t for t in tokens if not (rec_dir / f"{t}.npz").exists()]
    print(f"{len(tokens)} tokens, {len(todo)} to do", flush=True)
    data = DataLoader(_TokenFeatures(loader, agent.get_feature_builders(), todo), batch_size=a.batch_size,
                      num_workers=a.workers, collate_fn=_collate, worker_init_fn=_worker_init,
                      persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    t0, n = time.time(), 0
    with torch.inference_mode():
        for btok, feats in data:
            feats = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
            agent._loss.apply_aux_scales(model)
            pred = model(feats, run_aux=True)
            det = decode_detection_predictions(pred, max_predictions=100)
            mp = decode_map_predictions(pred, pc_range=map_pc_range, max_predictions=100)
            tau0 = pred["trajectory"].float().cpu().numpy()
            for i, t in enumerate(btok):
                with np.load(E0_REC / f"{t}.npz") as z:
                    gt = {k: z[k] for k in ("det_gt_boxes", "det_gt_labels", "map_gt_points", "map_gt_labels")}
                np.savez(rec_dir / f"{t}.npz", token=np.asarray(t),
                         det_pred_boxes=np.asarray(det[i]["boxes"], np.float32),
                         det_pred_scores=np.asarray(det[i]["scores"], np.float32),
                         det_pred_labels=np.asarray(det[i]["labels"], np.int64),
                         map_pred_points=np.asarray(mp[i]["points"], np.float32),
                         map_pred_scores=np.asarray(mp[i]["scores"], np.float32),
                         map_pred_labels=np.asarray(mp[i]["labels"], np.int64), tau0=tau0[i], **gt)
            n += len(btok)
            if n % 1000 < a.batch_size:
                print(f"  {n}/{len(todo)} {n / (time.time() - t0):.1f} tok/s", flush=True)
    meta = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                eval_cfg=str(EVAL_CFG), batch_size=a.batch_size, fp32=True, n_tokens=len(tokens),
                gt_source=str(E0_REC), sec=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
