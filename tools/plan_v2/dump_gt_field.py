"""The ground-truth corridor field for every navtest scene, on the PREDICTED field's grid.

This exists to test the one claim the corridor line rests on and has no evidence for: that making
the field more accurate makes the score go up.  Every arm so far confounds field quality with
everything else training changed -- `feas_gt` and `feas_pred` end with the SAME field error (0.669
vs 0.668 val L1) yet different gate gains, so arm-to-arm comparison cannot answer it.

The inference gate makes a clean test possible instead.  It is a selection rule over a finished
checkpoint, so the field can be SWAPPED while the model, the candidates, the offsets and the
ranking all stay fixed:

    Phi_alpha = alpha * GT + (1 - alpha) * predicted

alpha traces field error continuously from the model's own to zero, and nothing else moves.  If the
score rises monotonically in alpha, field accuracy causes score; the slope says what 0.1 m of field
error is worth, and alpha=1 is the ceiling that field improvement can ever reach.

Rasterised by the training target builder itself (DrivableFieldTargetBuilder), then averaged down
to the head's grid exactly as the grounding loss does (avg_pool2d), so "GT" here is the same object
the field was trained against -- not a second, subtly different ground truth.

    python tools/plan_v2/dump_gt_field.py --workers 10
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/readout"))
# BEFORE any navsim import: navsim.common.dataclasses reads NUPLAN_MAPS_ROOT at import time into a
# module constant, so importing it first captures None and every map query dies inside nuplan.
import _env  # noqa: E402,F401

_CTX = {}


def _cfg():
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    c = OmegaConf.load(ROOT / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    c.checkpoint_path = None
    # the field arms' settings: the dynamic field is per-step, and the target upsample sets the grid
    c.config.feas_weight = 0.15
    c.config.feas_dyn_weight = 1.0
    return instantiate(c.config, _convert_="all") if "_target_" in c.config else c.config


def _init():
    import torch
    from navsim.agents.para_ssr.feasibility import DrivableFieldTargetBuilder
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from _env import scene_filter, split_dirs

    cfg = _cfg()
    _CTX["cfg"] = cfg
    _CTX["builder"] = DrivableFieldTargetBuilder(cfg)
    logs_dir, blobs_dir = split_dirs("test")
    _CTX["loader"] = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest"),
                                 sensor_config=SensorConfig.build_no_sensors())
    _CTX["torch"] = torch


def _one(token: str):
    """-> (token, static [50,100], dyn [8,50,100]) on the head's grid, or (token, error)."""
    import torch.nn.functional as F
    try:
        scene = _CTX["loader"].get_scene_from_token(token)
        t = _CTX["builder"].compute_targets(scene)
        cfg = _CTX["cfg"]
        bw = int(cfg.bev_w)
        s = t["drivable_sdf_bev"].float()[None, None]                  # [1,1,H,W] fine grid
        r = s.shape[-1] // bw
        s = (F.avg_pool2d(s, r) if r > 1 else s)[0, 0]                 # same reduction as the loss
        out = [token, s.numpy().astype(np.float16), None]
        if "agent_sdf_bev" in t:
            d = t["agent_sdf_bev"].float()[None]                       # [1,T,H,W]
            rd = d.shape[-1] // bw
            out[2] = (F.avg_pool2d(d, rd) if rd > 1 else d)[0].numpy().astype(np.float16)
        return tuple(out)
    except Exception as exc:
        return token, repr(exc)[:160], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "work_dirs/eval_epdms/gt_field_navtest.npz"))
    a = ap.parse_args()

    z = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
    toks = [str(t) for t in z["tokens"]]
    if a.limit:
        toks = toks[: a.limit]
    print(f"{len(toks)} 장면", flush=True)

    stat, dyn, bad, t0 = {}, {}, [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for i, (tok, s, d) in enumerate(ex.map(_one, toks, chunksize=8), 1):
            if isinstance(s, str):
                bad.append((tok, s))
            else:
                stat[tok] = s
                if d is not None:
                    dyn[tok] = d
            if i % 1000 == 0:
                el = time.time() - t0
                print(f"{i}/{len(toks)}  {el / i:.3f} s/scene  남은 {(len(toks) - i) * el / i / 60:.0f}분",
                      flush=True)

    keep = [t for t in toks if t in stat]
    arr = np.stack([stat[t] for t in keep])
    out = {"tokens": np.asarray(keep), "drivable_sdf_gt": arr}
    if dyn:
        out["agent_sdf_gt"] = np.stack([dyn[t] for t in keep])
    np.savez_compressed(a.out, **out)
    print(f"\n저장 {a.out}  {arr.shape}  실패 {len(bad)}")
    for t, e in bad[:5]:
        print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
