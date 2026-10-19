"""refiner v4 fixes of review SSR/report 79 (verified 2026-10-16): near-stop candidates always read a station (F1),
the eval consistency checks pass the parents (F2), exact leave-one-out credit (F3), AWR after an fp16 autocast
forward (F8), evidence bins to 5.0 s (F6), specialist-teacher station keys (D(c)), and the parameter budget of the v4
critic (F7: = the CK2 critic it replaces).  CPU, synthetic inputs.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_review_fixes.py
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.ck.model import CKNet, save_ck
from navsim.agents.para_ssr.ck.v4 import V4Cfg
from navsim.agents.para_ssr.ck.v4 import corrector as CR
from navsim.agents.para_ssr.ck.v4 import losses as V4L
from navsim.agents.para_ssr.ck.v4.readout import I_DAC, I_NC, I_TTC, feasibility

HAZ = V4Cfg(readout="hazard", pref_head=True)
EVI = V4Cfg(readout="evidence", pref_head=True)


def _stop(S_end: float, t_stop: float = 2.0):
    """a draft that brakes to a stop at S_end [m] by t_stop [s] and stays there ([8, 3])."""
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    u = (t / t_stop).clamp(max=1.0)
    x = S_end * (1 - (1 - u) ** 2)
    return torch.stack([x, 0 * x, 0 * x], -1)


def _status(v0: float, n: int = 1):
    s = torch.zeros(n, 8)
    s[:, 1] = 1
    s[:, 4] = v0
    return s


# ----------------------------------------------------------------------------------------------- F1 near-stop reads
def test_near_stop_reads_and_learns():
    torch.manual_seed(0)
    net = CKNet("S", seed=0, v4=HAZ)
    net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    move = torch.stack([6 * t, 0 * t, 0 * t], -1)
    cand = torch.stack([_stop(0.4), _stop(0.5), _stop(1.4), move])[None]           # [1, 4, 8, 3]
    par = move[None, None].expand(1, 4, 8, 3)                                     # one shared parent path
    bev = torch.randn(1, 256, 50, 100)
    o = net(bev, cand, _status(6.0), decode=False, cand_parent=par)
    m = o["haz"]["haz_mask"][0]
    assert bool(m[:, 0].all())                                                    # station 0 always read
    assert int(m[0].sum()) == 1 and int(m[3].sum()) > 10                          # a 0.4 m stop reads bin 0 only
    # the stop dwells in its bin until the horizon; the moving draft passes through
    assert abs(float(o["haz"]["t_out"][0, 0, 0]) - 4.0) < 1e-6 and float(o["haz"]["t_out"][0, 3, 0]) < 0.5
    lg = o["score_logit"][0]
    assert float(lg[:, I_NC].max()) < 13.0                                        # no 'pass = 1 - 1e-6' ceiling
    # stopping at 0.5 m vs 1.4 m on the same parent pass (the old first-arrival read gave both the same few
    # stations) -> different reads, different scores
    assert float((lg[1, [I_NC, I_TTC, I_DAC]] - lg[2, [I_NC, I_TTC, I_DAC]]).abs().max()) > 1e-4
    # a failing near-stop label trains the station hazard heads
    F.binary_cross_entropy_with_logits(lg[0, I_NC], torch.tensor(0.0)).backward()
    g = net.v4_readout.h_nc[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    # the same 0.4 m stop is infeasible from 6 m/s but not from rest: the feasibility input differs
    e6 = feasibility(torch.cat([torch.zeros(1, 1), _stop(0.4)[None, :, 0]], 1), torch.tensor([6.0]))
    e0 = feasibility(torch.cat([torch.zeros(1, 1), _stop(0.4)[None, :, 0]], 1), torch.tensor([0.0]))
    assert float(e6[0, 2]) > 2.0 and float(e0[0, 2]) == 0.0 and float(e6[0, 1]) < -1.0


# ----------------------------------------------------------------------------------------------- F7 parameter budget
def _trainable(net) -> int:
    frozen = ("trunk.lon_head", "trunk.gate_head")          # CK2 as trained (train_ck2.FROZEN_MODULES, online2)
    return sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not n.startswith(frozen))


def test_param_budget():
    ck2 = _trainable(CKNet("S", seed=0))
    assert ck2 == 3_391_755
    for cfg in (HAZ, EVI):
        n = _trainable(CKNet("S", seed=0, v4=cfg))
        assert abs(n - ck2) / ck2 < 0.002, (cfg.readout, n, ck2)
    # the learned corrector A is a generator on top of the critic (report 75), outside the critic budget
    na = _trainable(CKNet("S", seed=0, v4=V4Cfg(readout="hazard", pref_head=True, corrector=True)))
    assert na > ck2 + 100_000


# ----------------------------------------------------------------------------------------------- F3 credit
def test_credit_exact_when_the_edit_dominates():
    # report 79's case: p_edit rounds to 1 in fp32; exact A_sel = f(S) - f(S \ edit) = 0.5 - 1.0, A = -0.75
    c = CR.credit(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.5]]), torch.ones(1, 2, dtype=torch.bool),
                  torch.tensor([[1]]), R_src=torch.tensor([[1.0]]))
    assert abs(float(c["A_sel"][0, 0]) + 0.5) < 1e-6 and abs(float(c["A"][0, 0]) + 0.75) < 1e-6
    # random sets vs a float64 brute force (several edits, not-ok columns)
    g = torch.Generator().manual_seed(0)
    s = torch.randn(4, 12, generator=g) * 2
    R = torch.rand(4, 12, generator=g)
    ok = torch.rand(4, 12, generator=g) > 0.2
    ok[:, 9:] = True
    cols = torch.tensor([[9, 10, 11]] * 4)
    c = CR.credit(s, R, ok, cols, R_src=torch.zeros(4, 3), tau=0.05)
    for b in range(4):
        sb = torch.where(ok[b], s[b].double(), torch.tensor(-1e4, dtype=torch.float64)) / 0.05
        Rb = torch.where(ok[b], R[b].double(), torch.zeros(12, dtype=torch.float64))
        f_all = (torch.softmax(sb, 0) * Rb).sum()
        for i, col in enumerate(cols[b].tolist()):
            keep = torch.arange(12) != col
            f_wo = (torch.softmax(sb[keep], 0) * Rb[keep]).sum()
            assert abs(float(c["A_sel"][b, i]) - float(f_all - f_wo)) < 1e-5


# ----------------------------------------------------------------------------------------------- F8 AWR after amp
def test_awr_after_fp16_autocast_forward():
    from navsim.agents.para_ssr.ck.online2 import CKE2E2, student_forward2
    torch.manual_seed(0)
    net = CKNet("S", seed=0, v4=V4Cfg(readout="hazard", pref_head=True, corrector=True))
    t = torch.arange(1, 9) * 0.5
    B, KG, KE = 2, 3, 2
    src = torch.stack([6 * t, 0 * t, 0 * t], -1)[None, None].expand(B, KG, 8, 3).clone()
    src[:, 1, :, 1] += 0.3
    E = {"E_src_traj": src[:, :KE].clone(), "E_src_par": src[:, :KE].clone(), "E_u": 0.1 * torch.randn(B, KE, 12),
         "G_pdms": torch.rand(B, KG), "E_pdms": torch.rand(B, KE), "E_ok": torch.ones(B, KE, dtype=torch.bool),
         "E_src_pdms": torch.rand(B, KE)}
    S = {"E": E, "n_G": KG, "ok_lab": torch.ones(B, KG + KE, dtype=torch.bool)}
    ext = torch.cat([src, src[:, :KE]], 1)
    with torch.autocast("cpu", dtype=torch.float16):           # = online2.loss with amp on (CUDA there)
        out = student_forward2(net, torch.randn(B, 256, 50, 100), src, _status(6.0, B), extra=ext, cand_parent=src,
                               extra_parent=ext)
    assert out["scene"][0].dtype == torch.float16
    logs = {}
    l = CKE2E2._awr(None, net, out, S, out["extra_score_logit"].float(), logs)
    l.backward()
    assert torch.isfinite(l) and logs["v4/awr_n"] == B * KE
    assert sum(float(p.grad.abs().sum()) for p in net.corrector.parameters() if p.grad is not None) > 0


# ----------------------------------------------------------------------------------------------- F2 eval checks
def _fake_pred(B: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    sp = torch.linspace(2.0, 10.0, 256)
    hd = torch.linspace(-0.08, 0.08, 256)
    x = sp[:, None] * t[None] * torch.cos(hd[:, None] * t[None])
    y = sp[:, None] * t[None] * torch.sin(hd[:, None] * t[None])
    anchors = torch.stack([x, y, hd[:, None] * t[None]], -1)
    return {"plan_final_rewards": torch.rand(B, 256, generator=g), "im_rewards": torch.rand(B, 256, generator=g),
            "sim_rewards": torch.rand(B, 5, 256, generator=g), "trajectory_anchors": anchors,
            "trajectory_offset": 0.05 * torch.randn(B, 256, 8, 3, generator=g),
            "bev_embed": torch.randn(B, 5000, 256, generator=g)}


@pytest.mark.parametrize("mode", ["hazard", "evidence", "corrector"])
def test_stage_infer_check_passes_for_v4_students(tmp_path, mode):
    from navsim.agents.para_ssr.ck.online2 import CKE2E2, CKE2E2Config
    from tools.ck.e2e2 import eval_e2e2 as E2
    B = 2
    kw = {"hazard": HAZ, "evidence": EVI, "corrector": V4Cfg(readout="hazard", pref_head=True, corrector=True)}[mode]
    torch.manual_seed(3)
    net = CKNet("S", seed=3, v4=kw, corridor_version=2).eval()
    P = E2.Paths(tmp_path / mode, "navtrain_val")
    for d in (P.ck_run, P.infer, P.packed):
        d.mkdir(parents=True, exist_ok=True)
    cfg = dict(arm="S", seed=3, v4=kw.to_dict(), corridor_version=2)
    save_ck(P.ck_run / "ckpt_last.pt", net, cfg)
    (P.ck_run / "config.json").write_text(json.dumps(cfg))
    pred = _fake_pred(B)
    status = _status(6.0, B)
    with torch.no_grad():
        r = CKE2E2(CKE2E2Config()).infer(net, {"status_feature": status}, pred)
    np.save(P.packed / "cand96.npy", r["ck2_cand96"].float().numpy())
    np.save(P.packed / "valid96.npy", r["ck2_valid96"].numpy())
    np.save(P.packed / "status.npy", status.numpy())
    bevg = pred["bev_embed"].transpose(1, 2).reshape(B, 256, 50, 100)
    np.savez(P.infer / "check_bev.npz", rows=np.arange(B), bev=bevg.numpy())
    for src, dst in E2.PRED_TO_INFER.items():
        np.save(P.infer / f"{dst}.npy", r[src].float().numpy())
    if mode == "corrector":
        for src, dst in E2.PRED_TO_EDIT.items():
            np.save(P.infer / f"{dst}.npy", r[src].numpy())
    res = E2.stage_infer_check(P, SimpleNamespace(gpus=""))
    assert res["pass"], res["max_abs_err"]
    if mode == "corrector":
        assert "edit_score_logit" in res["max_abs_err"] and "edit_traj" in res["max_abs_err"]


def test_teacher_r34_sanity_check_with_parents():
    from tools.ck.e2e2 import eval_teacher_navtest_r34 as R
    torch.manual_seed(0)
    net = CKNet("S", seed=0, v4=V4Cfg(readout="hazard")).eval()
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    cand = torch.stack([torch.stack([(5 + k) * t, 0.1 * k * t, 0 * t], -1) for k in range(16)])[None]
    var = cand[:, :, None].expand(1, 16, R.NVAR, 8, 3).reshape(1, 16 * R.NVAR, 8, 3).clone()
    var[..., 0] *= 0.9                                                       # slower 'variants' of their parents
    b = {"bev": torch.randn(1, 256, 50, 100), "cand": cand, "status": _status(6.0)}
    out = R.sanity_checks(net, b, var, torch.device("cpu"))
    assert out["fwd_score80_maxdiff"] < 1e-5 and out["fwd_score16_maxdiff"] < 1e-5


# ----------------------------------------------------------------------------------------------- D(c) specialist keys
def test_specialist_teacher_station_keys():
    from tools.ck import train_ck2 as TR
    torch.manual_seed(0)
    B, K = 2, 4
    haz = {k: torch.zeros(B, K, 48, requires_grad=True) for k in ("a_nc", "a_ttc", "a_dac")}
    t = torch.arange(48, dtype=torch.float32) * 0.1
    haz.update(haz_mask=torch.ones(B, K, 48, dtype=torch.bool), t_in=t.expand(B, K, 48),
               t_out=(t + 0.1).expand(B, K, 48))
    batch = {"anc_tidx": torch.full((B, K, 3), 10, dtype=torch.long), "y": torch.zeros(B, K, 5),
             "y_ok": torch.ones(B, K, dtype=torch.bool)}
    out = {"haz": haz}
    a = SimpleNamespace(lambda_station=0.5, lambda_ev=0.0, score_keys=["dac", "ep"])   # MAP teacher
    st = {}
    loss = TR.add_v4_losses(torch.zeros(()), st, out, batch, a)
    loss.backward()
    assert float(haz["a_dac"].grad.abs().sum()) > 0
    assert haz["a_nc"].grad is None or float(haz["a_nc"].grad.abs().sum()) == 0.0
    assert "st_bce_dac" in st and "st_bce_nc" not in st and "st_bce_ttc" not in st
    st2 = {}
    a2 = SimpleNamespace(lambda_station=0.5, lambda_ev=0.0, score_keys=["nc", "ttc", "ep", "comfort"])  # DET
    TR.add_v4_losses(torch.zeros(()), st2, {"haz": {k: v.detach() for k, v in haz.items()}}, batch, a2)
    assert "st_bce_nc" in st2 and "st_bce_ttc" in st2 and "st_bce_dac" not in st2


# ----------------------------------------------------------------------------------------------- F6 evidence horizon
def test_evidence_bins_reach_the_ttc_horizon():
    assert V4Cfg().ev_tbins == 10 and V4Cfg(readout="evidence").validate().ev_tbins == 10
    with pytest.raises(ValueError):
        V4Cfg(readout="evidence", ev_tbins=11).validate()
    # the TTC read of the last knot (t 4.0 s + 0.9 s) falls between bins 8 (4.5 s) and 9 (5.0 s), not clamped
    tb = (4.0 + 0.9) / 0.5 - 1.0
    assert 8 < tb < 10 - 1
