"""Shared helpers of the CK2 e2e model-side CPU tests (test_ck2_e2e_{config,teacher,loss,bevkd,callback}.py).

Not a test module (no test_ prefix).  Real inputs: the CK2 2-token fixture (NU40; packed navtrain_train rows 0 and 40000,
built by tests/test_ck2_e2e_data.make_ck2_fixture from the read-only CK1 fixture) and the 1-epoch CK2 smoke teachers
(D/ck2/smoke/smoke_ck2{T,M}_ddp2, train_ck2 runs with done.json).  v2 predictions are synthetic (random BEV / offsets /
rewards on the real 256 anchors).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
SMOKE_T = D / "ck2/smoke/smoke_ck2T_ddp2"
SMOKE_M = D / "ck2/smoke/smoke_ck2M_ddp2"
CK1_T = D / "train/ckT_p1"
ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"

_BATCH: Dict = {}


def have_inputs() -> bool:
    return (SMOKE_T / "done.json").is_file() and (SMOKE_M / "done.json").is_file() and (
        FIX2.is_file() or Path("/workspace/yongjae/ssd/yongjae_refiner/ck/phase2/impl-model/fixtures/real_b2.pt").is_file())


def batch() -> Dict:
    """{'tokens', 'rows', 'features', 'targets'} of the CK2 fixture (built once if absent)."""
    if not _BATCH:
        if not FIX2.is_file():
            sys.path.insert(0, str(REPO / "tests"))
            from test_ck2_e2e_data import make_ck2_fixture
            make_ck2_fixture(FIX2)
        _BATCH.update(torch.load(FIX2, map_location="cpu", weights_only=False))
    return _BATCH


def fresh(B: int = 2, rows=None):
    """(features, targets) clones of the fixture's first B tokens; rows: optional ck_row override (e.g. small rows for a
    small LabelStore2)."""
    d = batch()
    f = {k: v[:B].clone() for k, v in d["features"].items()}
    t = {k: v[:B].clone() for k, v in d["targets"].items()}
    if rows is not None:
        t["ck_row"] = torch.as_tensor(rows, dtype=torch.int64)
    return f, t


def fake_predictions(B: int, seed: int = 0, bev: Optional[torch.Tensor] = None, requires_grad: bool = True):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.from_numpy(np.load(ANCHORS)).float()
    off = 0.2 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    im = torch.softmax(torch.randn(B, 256, generator=g), -1)
    sim = torch.sigmoid(torch.randn(B, 5, 256, generator=g))
    if bev is None:
        bev = torch.randn(B, 5000, 256, generator=g)
        bev = (bev - bev.mean(-1, keepdim=True)) / bev.std(-1, keepdim=True)
        bev.requires_grad_(requires_grad)
    top = final.argmax(-1)
    traj = (anchors[None] + off)[torch.arange(B), top]
    return {"bev_embed": bev, "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anchors,
            "im_rewards": im, "sim_rewards": sim, "trajectory": traj}


def slice_pred(pred: Dict, sl) -> Dict:
    """predictions of a sub-batch (bev re-leafed with requires_grad)."""
    out = {}
    for k, v in pred.items():
        if k == "trajectory_anchors":
            out[k] = v
        elif k == "bev_embed":
            out[k] = v.detach()[sl].clone().requires_grad_(v.requires_grad)
        else:
            out[k] = v[sl].clone()
    return out


def cfg2(tmp: Path, **over) -> O.CKE2E2Config:
    d = {"enabled": True, "io_dir": str(Path(tmp) / "ck_e2e2"), "teacher_det_run": str(SMOKE_T),
         "teacher_map_run": str(SMOKE_M), "teacher_amp": False, "score_prior": "none", "grad_share_every": 0,
         "strict_rows": 0.0,
         # the smoke teachers predate --ep-target (config.json without the key = 'official') while the student
         # default was 'decoupled' (official since 2026-10-09): the runtime teacher ep_target check (TeacherPair2) is
         # off for these fixtures so tests may set either ep_target
         "teacher_ep_check": False,
         # no KD calibration files exist for the smoke teachers (kd_calib tests pass their own files)
         "kd_calib": {"enabled": False}}
    d.update(over)
    return O.CKE2E2Config.from_any(d)


def make_gen(io_dir: str, epoch: int, rows, n_rows: int, seed: int = 0, traj96: Optional[np.ndarray] = None,
             valid96: Optional[np.ndarray] = None):
    """A small (n_rows) CK2 generation with finite random labels for `rows` (written by write_generation_rows2)."""
    from navsim.agents.para_ssr.ck import e2e_data2 as D2

    rows = np.asarray(rows, np.int64)
    n = len(rows)
    rng = np.random.default_rng(seed)
    gen = D2.open_generation2(io_dir, epoch, n_rows=n_rows, mode="w+")
    if traj96 is None:
        a = np.load(ANCHORS).astype(np.float32)
        traj96 = np.stack([a[rng.choice(256, 96, replace=False)] for _ in range(n)]) + \
            0.05 * rng.standard_normal((n, 96, 8, 3)).astype(np.float32)
    if valid96 is None:
        valid96 = np.ones((n, 96), bool)
    labels = rng.random((n, 96, 9)).astype(np.float32).round(1)
    cand_ok = np.ones((n, 96), bool)
    D2.write_generation_rows2(gen, rows, traj96.astype(np.float32), labels, cand_ok, valid96)
    return traj96.astype(np.float32), labels, cand_ok


def all_grads_finite(module) -> bool:
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())
