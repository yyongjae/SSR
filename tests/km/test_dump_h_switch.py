"""KM_DUMP_H switch of CKE2E2.infer (listwise reranker cache) -- CPU only, synthetic inputs:

    S1  off: infer returns exactly the CK2 key set; on: the same keys with bitwise-equal values + ck2_h / ck2_h_lat /
        ck2_score_logit_lat
    S2  ck2_h is the score-head input: score_head(ck2_h) == ck2_score_logit
    S3  ck2_h_lat / ck2_score_logit_lat == a separate student_forward2 on the lateral trajectories (independent candidates)
    S4  student_forward2(return_h=False) has the old key set and values equal to return_h=True

    cd ~/kyungmin/SSR-ck2km && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \
      ~/miniconda3/envs/ssr/bin/python -m pytest -q -p no:cacheprovider tests/km/test_dump_h_switch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.online import bev_sgrid  # noqa: E402

ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
CK2_KEYS = {"ck2_cand_idx", "ck2_v2_final", "ck2_v2_im", "ck2_v2_sim", "ck2_cand96", "ck2_valid96", "ck2_score_logit",
            "ck2_w_lat", "ck2_e_lat", "ck2_lat_traj"}
NEW_KEYS = {"ck2_h", "ck2_h_lat", "ck2_score_logit_lat"}


def _inputs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.from_numpy(np.load(ANCHORS)).float()
    off = 0.2 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    bev = torch.randn(B, 5000, 256, generator=g)
    bev = (bev - bev.mean(-1, keepdim=True)) / bev.std(-1, keepdim=True)
    traj = (anchors[None] + off)[torch.arange(B), final.argmax(-1)]
    pred = {"bev_embed": bev, "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anchors,
            "im_rewards": torch.softmax(torch.randn(B, 256, generator=g), -1),
            "sim_rewards": torch.sigmoid(torch.randn(B, 5, 256, generator=g)), "trajectory": traj}
    status = torch.zeros(B, 8)
    status[:, 3] = 1.0                       # command one-hot
    status[:, 4] = 6.0 + torch.rand(B, generator=g)   # vx
    return {"status_feature": status}, pred


def _setup():
    c = O.CKE2E2Config.from_any({"enabled": True, "io_dir": "/nonexistent", "teacher_det_run": "x",
                                 "teacher_map_run": "y", "score_prior": "none", "kd_calib": {"enabled": False}})
    torch.manual_seed(0)
    st = O.build_student_ck2(c).eval()
    with torch.no_grad():
        for p in st.trunk.lat_head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return O.CKE2E2.__new__(O.CKE2E2), c, st


def _infer(ck, c, st, f, pred, on: bool):
    old = os.environ.pop("KM_DUMP_H", None)
    try:
        if on:
            os.environ["KM_DUMP_H"] = "1"
        ck.cfg = c
        return ck.infer(st, f, pred)
    finally:
        os.environ.pop("KM_DUMP_H", None)
        if old is not None:
            os.environ["KM_DUMP_H"] = old


def test_switch():
    ck, c, st = _setup()
    f, pred = _inputs()
    off = _infer(ck, c, st, f, pred, False)
    on = _infer(ck, c, st, f, pred, True)
    assert set(off) == CK2_KEYS                                           # S1
    assert set(on) == CK2_KEYS | NEW_KEYS
    for k in CK2_KEYS:
        assert torch.equal(off[k], on[k]), k
    B = off["ck2_score_logit"].shape[0]
    assert on["ck2_h"].shape == (B, 96, 384) and on["ck2_h_lat"].shape == (B, 96, 384)
    with torch.no_grad():                                                 # S2
        s = st.score_head(on["ck2_h"].reshape(B * 96, -1)).float().reshape(B, 96, -1)
    torch.testing.assert_close(s, on["ck2_score_logit"], rtol=0, atol=1e-5)
    with torch.no_grad():                                                 # S3
        o = O.student_forward2(st, bev_sgrid(pred["bev_embed"].float()), on["ck2_lat_traj"], f["status_feature"].float(),
                               return_h=True)
    torch.testing.assert_close(o["score_logit"], on["ck2_score_logit_lat"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(o["h_all"], on["ck2_h_lat"], rtol=1e-4, atol=1e-4)
    assert float((on["ck2_lat_traj"] - on["ck2_cand96"]).abs().max()) > 0  # the lateral trajectories differ


def test_forward_return_h():
    _, c, st = _setup()
    f, pred = _inputs(seed=1)
    cand = pred["trajectory"][:, None].repeat(1, 7, 1, 1) + 0.1 * torch.randn(2, 7, 8, 3)
    with torch.no_grad():
        a = O.student_forward2(st, bev_sgrid(pred["bev_embed"]), cand, f["status_feature"])
        b = O.student_forward2(st, bev_sgrid(pred["bev_embed"]), cand, f["status_feature"], return_h=True)
    assert set(b) - set(a) == {"h_all", "feat", "mem"}                    # S4
    for k in a:
        if k == "ego" or a[k] is None:
            assert k == "ego" or b[k] is None, k
        else:
            assert torch.equal(a[k], b[k]), k
