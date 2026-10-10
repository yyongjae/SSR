"""CK Phase 2 TeacherPair parity (contract model.tests 6): the online teachers (ckT_p1 + ckM_p1, AMP) on the Phase 1
r34 candidates of 2 real navtrain_train tokens reproduce the offline Phase 1 KD targets (kd_targets/phase1:
kd_score_prob, kd_score_prob_corr, kd_c_lon, kd_e_lat, kd_corr_traj).  GPU (fp16 autocast, as Phase 1) when CUDA is
visible (CUDA_VISIBLE_DEVICES must be in 0-3), else CPU fp32 (same tolerance)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from navsim.agents.para_ssr.ck import online as O

REPO = Path(__file__).resolve().parents[1]
CK = "/home/external-user/ssd/yongjae_refiner/ck"


def _device():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available() and vis not in ("", "-1"):
        if not set(vis.split(",")) <= {"0", "1", "2", "3"}:
            pytest.skip(f"CUDA_VISIBLE_DEVICES={vis}: only GPUs 0-3 may be used")
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_teacher_pair_matches_phase1_kd_targets():
    sys.path.insert(0, str(REPO / "tests"))
    from test_ck_e2e_model import real_batch

    d = real_batch()
    f, t = d["features"], d["targets"]
    dev = _device()
    tp = O.TeacherPair(f"{CK}/train/ckT_p1", f"{CK}/train/ckM_p1", "last", dev, amp=True)
    assert not any(p.requires_grad for n in tp.nets() for p in n.parameters())
    T = tp.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], t["ck_p1_cand"], f["status_feature"],
               rescore=True)
    ok = t["ck_p1_kd_ok"].to(dev) & t["ck_p1_ok"].to(dev)[:, None]
    assert bool(ok.all()) and bool(T["kd_ok"].all()) and bool(T["kd_ok_corr"].all())
    err = lambda a, b: float((a.float().to(dev) - b.float().to(dev))[ok].abs().max())  # noqa: E731
    e = {"prob": err(T["kd_score_prob"], t["ck_p1_kd_prob"]),
         "prob_corr": err(T["kd_score_prob_corr"], t["ck_p1_kd_prob_corr"]),
         "c_lon": err(T["kd_c_lon"], t["ck_p1_kd_c_lon"]), "e_lat": err(T["kd_e_lat"], t["ck_p1_kd_e_lat"]),
         "kd_corr": err(T["kd_corr"].flatten(2), t["ck_p1_kdc"].flatten(2))}
    print("TeacherPair vs Phase 1 max abs err", dev, e)
    assert e["prob"] <= 0.03 and e["prob_corr"] <= 0.03, e
    assert e["c_lon"] <= 0.05 and e["e_lat"] <= 0.05 and e["kd_corr"] <= 0.05, e
    # the teacher output has the decoder ranges of the KD space (mode A: c_lon <= 0, |e_lat| <= 2)
    assert float(T["kd_c_lon"].max()) <= 1e-6 and float(T["kd_e_lat"].abs().max()) <= 2.0 + 1e-6
    # no MAP teacher: DET everything (combine_teacher map=None)
    T1 = O.TeacherPair(f"{CK}/train/ckT_p1", "", "last", dev, amp=True).run(
        t["kd_bev_0"], t["kd_ok_0"], None, None, t["ck_p1_cand"], f["status_feature"], rescore=False)
    assert not bool(T1["kd_ok_corr"].any()) and bool(T1["kd_ok"].all())
    torch.testing.assert_close(T1["kd_c_lon"], T["kd_c_lon"])
