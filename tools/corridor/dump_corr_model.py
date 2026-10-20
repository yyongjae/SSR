#!/usr/bin/env python
"""km 10/15: dump a corridor-run checkpoint (corr30_*) for offline corridor selection.

  <out>/<split>/  traj.npy f16 [N, 256, 8, 3]  every candidate (anchor + offset), NAVSIM ego frame
                  sim.npy  f16 [N, 5, 256]     sim-reward sigmoids (nc, dac, ep, ttc, comfort)
                  im.npy   f16 [N, 256]        imitation softmax
                  final.npy f32 [N, 256]       plan_final_rewards (v2's own selection score)
                  v2traj.npy f32 [N, 8, 3]     v2's submitted trajectory
                  status.npy f32 [N, 4]        (vx, vy, ax, ay)
                  csdf.npy f16 [N, 100, 200]   predicted static SDF
                  cocc.npy u8 [N, 8, 100, 200] predicted swept occupancy probability * 255
                  done.npy bool [N]
  --det (km 10/16, motion-rendered corridor test A): only the det/motion head of the same forward, top-64 queries
  by max class sigmoid: det_box f16 [N, 64, 10] (metric x right / y fwd, logW, logL, z, logH, sin, cos, vx, vy),
  det_cls f16 [N, 64, 7] sigmoid, det_traj f16 [N, 64, 6, 8, 2] per-step offsets (SSR axes, 0.5 s),
  det_tcls f16 [N, 64, 6] mode logits; done file detdone_<i>.npy
Rows = <root>/packed/<split>/tokens.parquet.  Resumable; --part i/n shards the rows.

  CUDA_VISIBLE_DEVICES=0 $PY tools/corridor/dump_corr_model.py --run-dir <km_runs/corr30/RUN> --ckpt <ckpt> \
      --split navtrain_val --out <dir> [--part 0/2] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

ROOT = os.environ.get("KM_CORR_ROOT", "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root")  # parent of packed/<split>/tokens.parquet
SPEC = {"traj": (np.float16, (256, 8, 3)), "sim": (np.float16, (5, 256)), "im": (np.float16, (256,)),
        "final": (np.float32, (256,)), "v2traj": (np.float32, (8, 3)), "status": (np.float32, (4,)),
        "csdf": (np.float16, (100, 200)), "cocc": (np.uint8, (8, 100, 200))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cand-eps", type=float, default=None, help="km 10/20: override cand_select_eps (C runs)")
    ap.add_argument("--cand-lam", type=float, default=None, help="km 10/20: override cand_select_lam (C v3.1 smoothness)")
    ap.add_argument("--rows", default="", help="km 10/20: .npy of the row indices to dump (e.g. whole logs, for EC pairs)")
    ap.add_argument("--subset", type=int, default=0,
                    help="dump only the corr_select rows: sorted rng(0).choice(n, subset) (= its --n sample)")
    ap.add_argument("--planes", type=int, default=8, help="corridor_occ_planes of the run (cocc shape)")
    ap.add_argument("--fwd-rows", type=int, default=100, help="corridor rows: 100 (32 m) or 200 (64 m)")
    ap.add_argument("--route", action="store_true", help="the run predicts route channels (corridor_route_weight > 0)")
    ap.add_argument("--sigma", action="store_true", help="the run has the SDF sigma head (km 10/18): csig.npy")
    ap.add_argument("--det", action="store_true", help="dump only the det/motion head outputs (see the docstring)")
    ap.add_argument("--traj32", action="store_true", help="traj.npy in float32 (navtest submission; f16 is ~3 cm at 60 m)")
    ap.add_argument("--cand", action="store_true", help="km 10/19 (C): the candidate scorer outputs (cand_*.npy)")
    ap.add_argument("--no-fields", action="store_true", help="skip csdf / cocc (64 m x 49 planes: ~2 MB per token)")
    ap.add_argument("--select-rule", default=None,
                    help="override corridor_select_rule of the run config (km 10/17; default: the run's own)")
    a = ap.parse_args(argv)
    global SPEC
    if a.traj32:
        SPEC["traj"] = (np.float32, (256, 8, 3))
    if a.det:
        SPEC = {"det_box": (np.float16, (64, 10)), "det_cls": (np.float16, (64, 7)),
                "det_traj": (np.float16, (64, 6, 8, 2)), "det_tcls": (np.float16, (64, 6))}
    if not a.det:
        SPEC["msel"] = (np.int16, ())          # km 10/17: candidate the forward submitted (corridor_select_rule)
        if not a.no_fields:
            SPEC["cocc"] = (np.uint8, (a.planes, a.fwd_rows, 200))
            SPEC["csdf"] = (np.float16, (a.fwd_rows, 200))
    if a.cand and not a.det:   # km 10/19 (C v3): every scorer head (raw), the token heads, the expected official scores
        from navsim.agents.para_ssr.modules.candidate_scorer import HEADS
        SPEC.update({"cand_heads": (np.float16, (256, sum(HEADS.values()))), "cand_eff": (np.float16, (2,)),
                     "cand_exp_pdms1": (np.float16, (256,)), "cand_exp_epdms2": (np.float16, (256,)),
                     "cand_c1": (np.uint8, (256,))})
    if a.route and not a.det:
        SPEC["croute"] = (np.float16, (a.fwd_rows // 2, 100))
    if a.sigma and not a.det:
        SPEC["csig"] = (np.float16, (a.fwd_rows, 200))
    import pandas as pd
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.data import common as CM
    from tools.ck.data import dump_v2 as DV

    tdf = pd.read_parquet(f"{ROOT}/packed/{a.split}/tokens.parquet")
    n = len(tdf)
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    arr = {}
    for k, (dt, sh) in SPEC.items():
        f = d / f"{k}.npy"
        if not f.exists():
            np.lib.format.open_memmap(f, "w+", dt, (n,) + sh).flush()
        arr[k] = np.load(f, mmap_mode="r+")
    pi, pn = map(int, a.part.split("/"))
    fd = d / f"{'detdone' if a.det else 'done'}_{pi}.npy"
    done = np.load(fd) if fd.exists() else np.zeros(n, bool)
    pos = np.arange(n)[np.arange(n) % pn == pi]
    if a.subset:
        pos = np.intersect1d(pos, np.random.default_rng(0).choice(np.arange(n), a.subset, replace=False))
    if a.rows:
        pos = np.intersect1d(pos, np.load(a.rows))
    pos = pos[~done[pos]]
    if a.limit:
        pos = pos[: a.limit]
    print(f"{a.split} part {a.part}: todo {len(pos)}/{n}", flush=True)
    if not len(pos):
        return 0
    dev = torch.device("cuda:0")
    # the aux-metric protocol pins the 32 m range for comparable det/map mAP; this dump computes no mAP, so the
    # archived range is taken as is (64 m runs: pc_range y 0..64)
    cfg = OmegaConf.load(Path(a.run_dir) / "train/code/hydra/config.yaml").agent.config
    RAE.PROTOCOL_PC_RANGE = tuple(float(v) for v in cfg.pc_range)
    RAE.PROTOCOL_MAP_PC_RANGE = tuple(float(v) for v in cfg.map_pc_range)
    agent = RAE._build_agent(Path(a.run_dir) / "train/code/hydra/config.yaml", Path(a.ckpt), dev)
    agent.eval()
    if a.select_rule is not None:
        agent.config.corridor_select_rule = a.select_rule
        agent.para_ssr_model._config.corridor_select_rule = a.select_rule   # the model holds a copy (ckpt load)
    for k, v in (("cand_select_eps", a.cand_eps), ("cand_select_lam", a.cand_lam)):
        if v is not None:
            setattr(agent.config, k, v)
            setattr(agent.para_ssr_model._config, k, v)
    print(f"forward selection rule: {getattr(agent.para_ssr_model._config, 'corridor_select_rule', 'v2')}"
          f" (cand eps {getattr(agent.para_ssr_model._config, 'cand_select_eps', 0.02)},"
          f" lam {getattr(agent.para_ssr_model._config, 'cand_select_lam', 0.0)})", flush=True)
    sp = CM.split(a.split)
    toks = [str(t) for t in tdf.token.to_numpy()[pos]]
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    need = set(tdf.log.to_numpy()[pos])
    sf = replace(sf, tokens=list(toks), max_scenes=None)
    sf = replace(sf, log_names=sorted(set(sf.log_names or need) & need))
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]),
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    miss = set(toks) - set(loader.tokens)
    assert not miss, f"scene loader misses {len(miss)} tokens e.g. {sorted(miss)[:3]}"
    row_of = {t: int(p) for t, p in zip(toks, pos)}
    anchors = agent.para_ssr_model.pts_bbox_head.anchor_planner.trajectory_anchors.detach().float().to(dev)
    ncmd = int(agent.config.num_navi_cmd)
    dl = torch.utils.data.DataLoader(DV.FeatDS(loader, agent.get_feature_builders(), toks), batch_size=a.bs,
                                     num_workers=a.workers, collate_fn=DV.collate, pin_memory=True)
    t0, c = time.time(), 0
    for btok, feats, _ in dl:
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
              for k, v in feats.items()}
        r = np.array([row_of[t] for t in btok])
        if a.det:
            with torch.inference_mode():
                pr = agent.para_ssr_model(fg, run_aux=True)
            cls = pr["all_cls_scores"][-1].float().sigmoid()                                   # [b, Q, 7]
            top = cls.amax(-1).topk(64, dim=1).indices                                          # [b, 64]
            g = lambda x: torch.gather(x, 1, top.view(*top.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
            out = {"det_box": g(pr["all_bbox_preds"][-1].float()), "det_cls": g(cls),
                   "det_traj": g(pr["traj_preds"].float()), "det_tcls": g(pr["traj_cls_preds"].float())}
            for k, v in out.items():
                arr[k][r] = v.cpu().numpy().astype(SPEC[k][0])
            done[r] = True
            c += len(r)
            if c % 400 < len(r):
                for v in arr.values():
                    v.flush()
                np.save(fd, done)
                print(f"{c}/{len(pos)} {c / (time.time() - t0):.1f} tok/s", flush=True)
            continue
        with torch.inference_mode():
            pr = agent(fg)
        traj = anchors[None] + pr["trajectory_offset"].float()
        out = {"traj": traj, "sim": pr["sim_rewards"], "im": pr["im_rewards"], "final": pr["plan_final_rewards"],
               "v2traj": pr.get("trajectory_v2", pr["trajectory"]), "status": fg["status_feature"][:, ncmd:ncmd + 4],
               "msel": pr.get("plan_select_index", pr["plan_final_rewards"].argmax(1))}
        if not a.no_fields:
            out.update({"csdf": pr["corridor_sdf"], "cocc": torch.sigmoid(pr["corridor_occ"].float()) * 255})
        if a.cand:
            from navsim.agents.para_ssr.modules.candidate_scorer import HEADS, expected_official
            ex = expected_official(pr, pr["cand_states"])
            heads = torch.cat([pr["cand_" + nm].float().reshape(*pr["cand_prog"].shape, -1) for nm in HEADS], -1)
            out.update({"cand_heads": heads, "cand_eff": torch.stack([pr["cand_eff1"], pr["cand_eff2"]], -1),
                        "cand_exp_pdms1": ex["pdms1"], "cand_exp_epdms2": ex["epdms2"], "cand_c1": ex["c1"]})
        if a.route:
            out["croute"] = pr["corridor_route_s"]
        if a.sigma:
            out["csig"] = pr["corridor_sdf_sigma"]
        for k, v in out.items():
            v = v.float().cpu().numpy()
            arr[k][r] = np.rint(v).astype(np.uint8) if k == "cocc" else v.astype(SPEC[k][0])
        done[r] = True
        c += len(r)
        if c % 400 < len(r):
            for v in arr.values():
                v.flush()
            np.save(fd, done)
            print(f"{c}/{len(pos)} {c / (time.time() - t0):.1f} tok/s", flush=True)
    for v in arr.values():
        v.flush()
    np.save(fd, done)
    print(f"done {c} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
