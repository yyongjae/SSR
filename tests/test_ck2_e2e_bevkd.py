"""CK2 e2e T10: BEV-KD arm (ck/bev_kd_arm.py) and its wiring in CKE2E2.loss.  CPU only (gloo DDP for the collective):
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_bevkd.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/bevkd

GradScale (BEV gradient = lam x unit gradient, adapter gradient independent of lam, lam 0 -> no BEV gradient),
RatioController math (ratio, cap, start_mb, skip rules, state round trip), TeacherRatioControllers (one controller per
teacher, per-teacher overrides, state round trip incl. the old single-controller format, teacher-set mismatch refused),
BEVKDArm (teacher keys, z-score from the teacher runs incl. the MAP run's norm_map.npz, strict state dict), two teachers
(det + map, user 2026-10-08): BEV gradient = sum_t lam_t dL_t/dBEV (autograd against the plain per-teacher losses +
finite differences), adapters independent of lam, map-missing tokens (kd_ok_1 False) masked and counted, [det] alone
equal to the old single-weight formula; measurement only on the v2 'gnorm/plan' micro-batches with the new lam_t
applied from the next one, per teacher; the term returned inside the CK loss; controller state through the Lightning
callback-state functions; and a 2-process gloo DDP run with [det, map] (student + arm in DDP, find_unused_parameters
False, warm-up and on-policy, no MAP BEV on rank 1): identical per-teacher controllers / lam on both ranks after the
all-reduce, the rank-0 state restores every rank exactly, identical finite gradients on every trainable parameter.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import bev_kd_arm as A  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")


# ----------------------------------------------------------------------------------------------- GradScale
def test_gradscale():
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(3, 4)
    for s in (1.0, 0.3, 0.0):
        x.grad = None
        (A.GradScale.apply(x, s) * w).sum().backward()
        torch.testing.assert_close(x.grad, s * w)
    y = A.GradScale.apply(x, 0.5)
    assert torch.equal(y, x)


# ----------------------------------------------------------------------------------------------- RatioController
def test_ratio_controller_math_and_state():
    rc = A.RatioController(ratio=0.1, cap=0.25, m=0.9, start_mb=100)
    assert rc.weight(10 ** 6) == 0.0                                    # no measurement yet
    for g, ref in ((float("nan"), 1.0), (1.0, float("inf")), (0.0, 1.0), (1e-13, 1.0), (1.0, 0.0), (1.0, -1.0)):
        rc.update(g, ref)
    assert rc.n_skip == 6 and rc.n == 0 and rc.log_r is None
    rc.update(2.0, 1.0)                                                 # r = 0.5
    assert rc.weight(99) == 0.0                                         # before start_mb
    assert rc.weight(100) == pytest.approx(min(0.1 * 0.5, 0.25 * 0.5))
    rc.update(1.0, 1.0)                                                 # r = 1: log_r = 0.9 log 0.5
    assert rc.log_r == pytest.approx(0.9 * math.log(0.5)) and rc.r_last == 1.0
    assert rc.weight(100) == pytest.approx(0.1 * 0.5 ** 0.9)
    big = A.RatioController(ratio=0.5, cap=0.25, m=0.9, start_mb=0)
    big.update(1.0, 1.0)
    assert big.weight(0) == pytest.approx(0.25)                         # cap * r_last binds
    wm = A.RatioController(ratio=0.1, cap=0.25, start_mb=0, w_max=0.01)
    wm.update(1.0, 1.0)
    assert wm.weight(0) == pytest.approx(0.01)
    rc2 = A.RatioController(ratio=0.1, cap=0.25, m=0.9, start_mb=100)
    rc2.load_state_dict(rc.state_dict())
    assert rc2.state_dict() == rc.state_dict() and rc2.weight(100) == rc.weight(100)
    for bad in ({"ratio": 0.0}, {"cap": -1.0}, {"m": 1.0}, {"w_max": 0.0}):
        with pytest.raises(ValueError):
            A.RatioController(**bad)


def test_teacher_ratio_controllers():
    tc = A.TeacherRatioControllers(["det", "map"], ratio=0.1, cap=0.25, m=0.9, start_mb=10,
                                   per_teacher={"map": {"ratio": 0.2, "cap": None}})
    assert tc.teachers == ("det", "map") and tc["det"].ratio == 0.1 and tc["map"].ratio == 0.2 \
        and tc["map"].cap == 0.25
    assert tc.weights(100) == {"det": 0.0, "map": 0.0}
    tc.update({"det": 2.0, "map": 0.0}, 1.0)                           # map skipped (zero unit gradient)
    assert tc["det"].n == 1 and tc["map"].n == 0 and tc["map"].n_skip == 1
    assert tc.weights(9) == {"det": 0.0, "map": 0.0}                     # before start_mb
    assert tc.weights(10) == {"det": pytest.approx(min(0.1 * 0.5, 0.25 * 0.5)), "map": 0.0}
    tc.update({"det": 2.0, "map": 4.0}, 1.0)
    w = tc.weights(10)
    assert w["map"] == pytest.approx(min(0.2 * 0.25, 0.25 * 0.25)) and w["det"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="no unit gradient"):
        tc.update({"det": 1.0}, 1.0)
    sd = tc.state_dict()
    assert set(sd) == {"det", "map"} and sd["det"] == tc["det"].state_dict()
    tc2 = A.TeacherRatioControllers(["det", "map"], start_mb=10, per_teacher={"map": {"ratio": 0.2}})
    tc2.load_state_dict(sd)
    assert tc2.state_dict() == sd and tc2.weights(10) == tc.weights(10)
    for bad in ({"det": sd["det"]}, {"det": sd["det"], "map": sd["map"], "x": sd["map"]}):
        with pytest.raises(ValueError, match="teachers"):
            tc2.load_state_dict(bad)
    # old single-controller state (log_r / r_last / n / n_skip) loads into a [det] arm only
    one = A.TeacherRatioControllers(["det"], start_mb=0)
    one.load_state_dict(tc["det"].state_dict())
    assert one["det"].state_dict() == tc["det"].state_dict()
    with pytest.raises(ValueError, match="single-controller"):
        tc2.load_state_dict(tc["det"].state_dict())
    for kw in ({"teachers": []}, {"teachers": ["det", "det"]}, {"teachers": ["bev"]},
               {"teachers": ["det"], "per_teacher": {"map": {"ratio": 0.2}}},
               {"teachers": ["det"], "per_teacher": {"det": {"m": 0.5}}}):
        with pytest.raises(ValueError):
            A.TeacherRatioControllers(**kw)
    # from the config (ratio_<t> / cap_<t> overrides; null = shared)
    b = TL.cfg2(Path("/nonexistent"), bev_kd={"enabled": True, "teachers": ["det", "map"], "cap_map": 0.5}).bev_kd
    cc = A.controllers_from_cfg(b)
    assert cc.teachers == ("det", "map") and cc["map"].cap == 0.5 and cc["det"].cap == 0.25 \
        and cc["map"].ratio == cc["det"].ratio == 0.1 and cc["det"].start_mb == 10600


# ----------------------------------------------------------------------------------------------- arm module
def _arm_cfg(tmp_path, **bev):
    b = {"enabled": True}
    b.update(bev)
    return TL.cfg2(tmp_path, bev_kd=b)


def test_arm_module_gradients_and_state(tmp_path):
    from navsim.agents.para_ssr.ck.bev_kd import norm_from_run
    c = _arm_cfg(tmp_path, teachers=["det", "map"], init="identity")
    arm = A.build_bev_kd_arm(c)
    assert arm.teachers == ("det", "map") and arm.keys["det"] == ("kd_bev_0", "kd_ok_0") \
        and arm.keys["map"] == ("kd_bev_1", "kd_ok_1")
    m, s = norm_from_run(TL.SMOKE_T, "det")
    assert np.allclose(arm.kd.zscore["det"].mean.reshape(-1).numpy(), m) and \
        np.allclose(arm.kd.zscore["det"].std.reshape(-1).numpy(), s)
    assert set(arm.state_dict()) == {f"kd.{a}.{t}.{p}" for t in ("det", "map")
                                     for a, p in (("adapters", "proj.weight"), ("adapters", "proj.bias"),
                                                  ("zscore", "mean"), ("zscore", "std"))}
    mm, sm = norm_from_run(TL.SMOKE_M, "map")
    assert (Path(TL.SMOKE_M) / "norm_map.npz").is_file() and np.allclose(arm.kd.zscore["map"].mean.reshape(-1).numpy(), mm) \
        and np.allclose(arm.kd.zscore["map"].std.reshape(-1).numpy(), sm)
    _, t = TL.fresh(2)
    g = torch.Generator().manual_seed(0)
    bev0 = torch.randn(2, 5000, 256, generator=g)
    res = {}
    for lam in (1.0, 0.2, 0.0):
        arm.zero_grad()
        bev = bev0.clone().requires_grad_(True)
        out = arm(bev, t, lam)
        torch.testing.assert_close(out["term"], arm.adapter_weight * (out["loss/det"] + out["loss/map"]))
        torch.testing.assert_close(out["loss"], 0.5 * (out["loss/det"] + out["loss/map"]))
        unit = A.BEVKDArm.unit_grad_norms(out, arm.teachers)
        out["term"].backward()
        res[lam] = (bev.grad.clone(), [p.grad.clone() for p in arm.parameters()], unit)
    torch.testing.assert_close(res[0.2][0], 0.2 * res[1.0][0], rtol=1e-5, atol=1e-12)
    assert float(res[0.0][0].abs().max()) == 0.0
    for a, b in zip(res[1.0][1], res[0.0][1]):
        torch.testing.assert_close(a, b)                                            # adapter grad independent of lam
        assert float(b.abs().sum()) > 0
    # adapter_weight 2: the adapter step doubles, the BEV gradient is still lam x unit
    c2 = _arm_cfg(tmp_path, teachers=["det", "map"], init="identity", adapter_weight=2.0)
    arm2 = A.build_bev_kd_arm(c2)
    bev = bev0.clone().requires_grad_(True)
    out = arm2(bev, t, 1.0)
    out["term"].backward()
    torch.testing.assert_close(bev.grad, res[1.0][0], rtol=1e-5, atol=1e-12)
    for a, b in zip(arm2.parameters(), res[1.0][1]):
        torch.testing.assert_close(a.grad, 2.0 * b, rtol=1e-5, atol=1e-12)
    # zero init: no BEV gradient at step 0 (unit grad 0 -> the controller skips the measurement)
    z = A.build_bev_kd_arm(_arm_cfg(tmp_path))
    assert z.teachers == ("det",)
    out = z(bev0.clone().requires_grad_(True), t, 1.0)
    assert A.BEVKDArm.unit_grad_norms(out, z.teachers) == {"det": 0.0}
    assert A.bev_kd_parameters(type("Ag", (), {"ck_bev_kd": z})) == list(z.parameters())


def _plain_unit_grads(arm, bev0, t):
    """{t: dL_t / dbev} of the plain per-teacher losses (BEVFeatureKD on bev itself, no GradScale)."""
    from navsim.agents.para_ssr.ck.bev_kd import teacher_inputs
    tb, ok = teacher_inputs(t, arm.keys, arm.teachers)
    res = {}
    for name in arm.teachers:
        bev = bev0.clone().requires_grad_(True)
        o = arm.kd(bev, tb, ok)
        res[name] = torch.autograd.grad(o[f"loss/{name}"], bev, allow_unused=True)[0]
        res[name] = torch.zeros_like(bev0) if res[name] is None else res[name]
    return res


def _perturbed_arm(tmp_path, **bev):
    arm = A.build_bev_kd_arm(_arm_cfg(tmp_path, **bev))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():                      # different non-zero adapters, so both BEV gradients are measurable
        for i, a in enumerate(arm.kd.adapters.values()):
            a.proj.weight.add_((0.05 + 0.05 * i) * torch.randn(256, 256, generator=g))
    return arm


def test_two_teachers_per_teacher_lambda_bev_gradient(tmp_path):
    arm = _perturbed_arm(tmp_path, teachers=["det", "map"], adapter_weight=2.0)
    _, t = TL.fresh(2)
    bev0 = torch.randn(2, 5000, 256, generator=torch.Generator().manual_seed(1))
    plain = _plain_unit_grads(arm, bev0, t)
    assert all(float(v.norm()) > 0 for v in plain.values())
    runs = {}
    for lam in ({"det": 1.0, "map": 0.0}, {"det": 0.0, "map": 1.0}, {"det": 0.3, "map": 2.5}):
        arm.zero_grad()
        bev = bev0.clone().requires_grad_(True)
        out = arm(bev, t, lam)
        unit = A.BEVKDArm.unit_grad_norms(out, arm.teachers)
        for name in arm.teachers:                                       # unit = ||dL_t/dBEV|| whatever lam is
            assert unit[name] == pytest.approx(float(plain[name].norm()), rel=1e-5)
            assert float(out[f"lam/{name}"]) == pytest.approx(lam[name], rel=1e-7)
        out["term"].backward()
        want = lam["det"] * plain["det"] + lam["map"] * plain["map"]
        torch.testing.assert_close(bev.grad, want, rtol=1e-5, atol=1e-10)
        runs[tuple(lam.values())] = [p.grad.clone() for p in arm.parameters()]
    a, b, c = runs.values()
    for x, y, z in zip(a, b, c):                                        # adapter step independent of every lam_t
        torch.testing.assert_close(x, y)
        torch.testing.assert_close(x, z)
    # each adapter learns from its own teacher only, at adapter_weight (= 2 x its plain loss gradient)
    from navsim.agents.para_ssr.ck.bev_kd import teacher_inputs
    tb, ok = teacher_inputs(t, arm.keys, arm.teachers)
    o = arm.kd(bev0, tb, ok)
    for name in arm.teachers:
        ga = torch.autograd.grad(o[f"loss/{name}"], list(arm.kd.adapters[name].parameters()))
        other = [n for n in arm.teachers if n != name][0]
        gx = torch.autograd.grad(o[f"loss/{other}"], list(arm.kd.adapters[name].parameters()), allow_unused=True)
        assert all(g is None for g in gx)
        idx = [i for i, p in enumerate(arm.parameters()) if any(p is q for q in arm.kd.adapters[name].parameters())]
        for i, g in zip(idx, ga):
            torch.testing.assert_close(a[i], 2.0 * g, rtol=1e-5, atol=1e-9)
    # central finite differences of the plain per-teacher loss (the module runs in float32) along a unit direction
    # mixing the gradient direction and a random one: (L(b + e d) - L(b - e d)) / 2e == <dL/db, d>
    r = torch.randn(2, 5000, 256, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        for name in arm.teachers:
            gdir = plain[name] / plain[name].norm()
            d = gdir + r / r.norm()
            d = d / d.norm()
            f = lambda x: float(arm.kd(x, tb, ok)[f"loss/{name}"].double())  # noqa: E731
            eps = 1.0
            fd = (f(bev0 + eps * d) - f(bev0 - eps * d)) / (2 * eps)
            ad = float((plain[name].double() * d.double()).sum())
            assert fd == pytest.approx(ad, rel=2e-3), (name, fd, ad)


def test_det_only_equals_old_single_weight_formula(tmp_path):
    """[det]: term / BEV gradient / adapter gradient / unit norm == the old arm (adapter_weight * L(GradScale(bev,
    lam / adapter_weight)), L = BEVFeatureKD loss = stack([L_det]).mean()) bit for bit."""
    arm = _perturbed_arm(tmp_path, teachers=["det"], adapter_weight=1.5)
    _, t = TL.fresh(2)
    t["kd_ok_0"][1] = False
    from navsim.agents.para_ssr.ck.bev_kd import teacher_inputs
    tb, ok = teacher_inputs(t, arm.keys, arm.teachers)
    bev0 = torch.randn(2, 5000, 256, generator=torch.Generator().manual_seed(2))
    for lam in (0.0, 0.7, 3.0):
        arm.zero_grad()
        b_new = bev0.clone().requires_grad_(True)
        out = arm(b_new, t, {"det": lam})
        u_new = A.BEVKDArm.unit_grad_norms(out, arm.teachers)["det"]
        out["term"].backward()
        g_new = [p.grad.clone() for p in arm.parameters()]
        arm.zero_grad()
        b_old = bev0.clone().requires_grad_(True)
        x = A.GradScale.apply(b_old, lam / arm.adapter_weight)
        ref = arm.kd(x, tb, ok)
        term_old = arm.adapter_weight * ref["loss"]
        u_old = float(torch.autograd.grad(ref["loss"], x, retain_graph=True)[0].norm())
        term_old.backward()
        assert torch.equal(out["term"].detach(), term_old.detach()) and u_new == u_old
        assert torch.equal(b_new.grad, b_old.grad)
        assert all(torch.equal(a, p.grad) for a, p in zip(g_new, arm.parameters()))


def test_map_missing_tokens_masked_and_counted(tmp_path):
    c = _arm_cfg(tmp_path, teachers=["det", "map"], start_mb=0)
    arm = _perturbed_arm(tmp_path, teachers=["det", "map"], start_mb=0)
    _, t = TL.fresh(2)
    bev0 = torch.randn(2, 5000, 256, generator=torch.Generator().manual_seed(4))
    full = _plain_unit_grads(arm, bev0, t)
    t1 = {k: v.clone() for k, v in t.items()}
    t1["kd_ok_1"][0] = False                                           # token 0: no ReSMap BEV
    t1["kd_bev_1"][0] = float("nan")                                   # whatever is stored there must not leak
    part = _plain_unit_grads(arm, bev0, t1)
    assert float(part["map"][0].abs().max()) == 0.0 and float(part["map"][1].abs().max()) > 0
    torch.testing.assert_close(part["det"], full["det"])               # DET untouched by the MAP mask
    torch.testing.assert_close(part["map"][1], 2.0 * full["map"][1], rtol=1e-5, atol=1e-10)   # mean over 1 ok token
    out = arm(bev0.clone().requires_grad_(True), t1, 1.0)
    assert int(out["n_ok/map"]) == 1 and int(out["n_ok/det"]) == 2 and torch.isfinite(out["term"])
    # through CKE2E2: per-teacher ok_frac logged, missing tokens counted, MAP-free mb -> map controller skips
    ck = O.CKE2E2(c)
    torch.manual_seed(0)
    st = O.build_student_ck2(c)
    f, _ = TL.fresh(2)
    t2 = {k: v.clone() for k, v in t.items()}
    t2["kd_ok_1"][:] = False
    for i, tt in enumerate((t1, t2)):
        _, logs = ck.loss(st, f, tt, TL.fake_predictions(2, seed=i), v2_logs={"gnorm/plan": torch.tensor(0.5)},
                          bev_kd=arm)
        assert float(logs["bevkd/map/ok_frac"]) == (0.5 if i == 0 else 0.0)
        assert float(logs["bevkd/det/ok_frac"]) == 1.0 and torch.isfinite(logs["bevkd/term"])
    assert ck.cum["bevkd_tok"] == 4 and ck.cum["bevkd_miss_map"] == 1 + 2 and ck.cum["bevkd_miss_det"] == 0
    assert float(logs["bevkd/map/g_kd_unit"]) == 0.0 and float(logs["bevkd/map/loss"]) == 0.0
    assert ck.bev_ctrl["det"].n == 2 and ck.bev_ctrl["map"].n == 1 and ck.bev_ctrl["map"].n_skip == 1
    # all MAP missing: the map adapter is still in the graph with a zero gradient (DDP find_unused_parameters False)
    arm.zero_grad()
    out = arm(bev0.clone().requires_grad_(True), t2, 1.0)
    out["term"].backward()
    assert all(p.grad is not None and float(p.grad.abs().max()) == 0.0 for p in arm.kd.adapters["map"].parameters())
    assert any(float(p.grad.abs().max()) > 0 for p in arm.kd.adapters["det"].parameters())


def test_loss_wiring_measurement_and_next_mb_lambda(tmp_path):
    c = _arm_cfg(tmp_path, start_mb=0, init="identity")
    ck = O.CKE2E2(c)
    arm = A.build_bev_kd_arm(c)
    torch.manual_seed(0)
    st = O.build_student_ck2(c)
    f, t = TL.fresh(1)
    lam_log, ctrl_n = [], []
    for i, v2l in enumerate((None, {"gnorm/plan": torch.tensor(0.5)}, None, {"gnorm/plan": torch.tensor(0.5)})):
        pred = TL.fake_predictions(1, seed=i)
        v2 = (pred["bev_embed"] ** 2).mean()
        loss, logs = ck.loss(st, f, t, pred, v2_loss=v2, v2_logs=v2l, bev_kd=arm)
        lam_log.append(float(logs["bevkd/det/lam"]))
        ctrl_n.append(ck.bev_ctrl["det"].n)
        assert ("bevkd/det/g_kd_unit" in logs) == (v2l is not None) == ("bevkd/g_plan" in logs)
        assert float(loss) == pytest.approx(float(logs["ck2/loss"]) + float(logs["bevkd/term"]), rel=1e-5)
        for k in ("bevkd/det/loss", "bevkd/det/cos", "bevkd/det/fve", "bevkd/det/w_dev", "bevkd/det/ok_frac",
                  "bevkd/det/lam", "bevkd/term"):
            assert k in logs and torch.isfinite(logs[k]), k
        assert not any(k.startswith("bevkd/map/") for k in logs)
    assert ctrl_n == [0, 1, 1, 2]
    assert lam_log[0] == 0.0 and lam_log[1] == 0.0                     # measured on mb 1, applied from mb 2
    g_unit = float(logs["bevkd/det/g_kd_unit"])
    assert lam_log[2] > 0 and float(logs["bevkd/det/ratio_now"]) == pytest.approx(lam_log[3] * g_unit / 0.5, rel=1e-4)
    # start_mb: no BEV pull before it, while the controller keeps measuring
    c2 = _arm_cfg(tmp_path, start_mb=10600, init="identity")
    ck2 = O.CKE2E2(c2)
    _, logs = ck2.loss(st, f, t, TL.fake_predictions(1, 0), v2_logs={"gnorm/plan": torch.tensor(0.5)},
                       bev_kd=A.build_bev_kd_arm(c2))
    assert float(logs["bevkd/det/lam"]) == 0.0 and ck2.bev_ctrl["det"].n == 1
    # grad share includes the arm's BEV gradient
    c3 = _arm_cfg(tmp_path, start_mb=0, init="identity")
    c3.grad_share_every = 1
    ck3 = O.CKE2E2(c3)
    ck3.bev_ctrl.update({"det": 1.0}, 1.0)
    pred = TL.fake_predictions(1, seed=7)
    _, logs = ck3.loss(st, f, t, pred, v2_loss=(pred["bev_embed"] ** 2).mean(), bev_kd=A.build_bev_kd_arm(c3))
    assert float(logs["gnorm/bev_bevkd"]) > 0
    gs = float(logs["gnorm/bev_ck"]) / (float(logs["gnorm/bev_v2"]) + float(logs["gnorm/bev_ck"]) +
                                         float(logs["gnorm/bev_bevkd"]))
    assert float(logs["ck2/gshare"]) == pytest.approx(gs, rel=1e-5)
    assert float(logs["gnorm/bev_bevkd_det"]) == pytest.approx(float(logs["gnorm/bev_bevkd"]), rel=1e-5)
    # state round trip through CKE2E2
    sd = ck.state_dict()
    assert set(sd["bev_ctrl"]) == {"det"}
    ck4 = O.CKE2E2(c)
    ck4.load_state_dict(sd)
    assert ck4.bev_ctrl.state_dict() == ck.bev_ctrl.state_dict()


def test_two_teacher_loss_wiring_per_teacher_controllers(tmp_path):
    c = _arm_cfg(tmp_path, teachers=["det", "map"], start_mb=0, ratio_map=0.2)
    c.grad_share_every = 1
    ck = O.CKE2E2(c)
    arm = _perturbed_arm(tmp_path, teachers=["det", "map"], start_mb=0)
    torch.manual_seed(0)
    st = O.build_student_ck2(c)
    f, t = TL.fresh(1)
    lams, logs_all = [], []
    for i, v2l in enumerate(({"gnorm/plan": torch.tensor(0.5)}, None, {"gnorm/plan": torch.tensor(0.8)}, None)):
        pred = TL.fake_predictions(1, seed=i)
        v2 = (pred["bev_embed"] ** 2).mean()
        loss, logs = ck.loss(st, f, t, pred, v2_loss=v2, v2_logs=v2l, bev_kd=arm)
        lams.append({n: float(logs[f"bevkd/{n}/lam"]) for n in ("det", "map")})
        logs_all.append(logs)
        assert float(loss) == pytest.approx(float(logs["ck2/loss"]) + float(logs["bevkd/term"]), rel=1e-5)
        for n in ("det", "map"):
            for k in ("loss", "lam", "ok_frac", "cos", "fve", "w_dev"):
                assert f"bevkd/{n}/{k}" in logs and torch.isfinite(logs[f"bevkd/{n}/{k}"]), (n, k)
            assert (f"bevkd/{n}/g_kd_unit" in logs) == (v2l is not None)
            assert (float(logs[f"gnorm/bev_bevkd_{n}"]) == 0.0) == (lams[-1][n] == 0.0)
    assert lams[0] == {"det": 0.0, "map": 0.0}                           # measured on mb 0, applied from mb 1
    g0 = {n: float(logs_all[0][f"bevkd/{n}/g_kd_unit"]) for n in ("det", "map")}
    assert g0["det"] > 0 and g0["map"] > 0 and g0["det"] != pytest.approx(g0["map"], rel=1e-3)
    for n, ratio in (("det", 0.1), ("map", 0.2)):                      # each lam_t from its own g_t (ratio_map 0.2)
        assert lams[1][n] == pytest.approx(min(ratio, 0.25) * 0.5 / g0[n], rel=1e-6)
        assert lams[1][n] == lams[2][n]
        g2 = float(logs_all[2][f"bevkd/{n}/g_kd_unit"])
        assert float(logs_all[2][f"bevkd/{n}/ratio_now"]) == pytest.approx(lams[2][n] * g2 / 0.8, rel=1e-6)
        rc = A.RatioController(ratio=ratio, cap=0.25, m=0.9, start_mb=0)   # = an independent controller per teacher
        rc.update(g0[n], 0.5)                                            # (logs are float32: compare to 1e-6)
        rc.update(g2, 0.8)
        sc, sr = ck.bev_ctrl[n].state_dict(), rc.state_dict()
        assert lams[3][n] == pytest.approx(rc.weight(3), rel=1e-6) and sc["n"] == sr["n"] == 2 and sc["n_skip"] == 0
        assert sc["log_r"] == pytest.approx(sr["log_r"], rel=1e-6) and sc["r_last"] == pytest.approx(sr["r_last"],
                                                                                                    rel=1e-6)
    # the weighted per-teacher BEV gradients: lam_t * |dL_t/dBEV| (mb 2, both lam > 0)
    for n in ("det", "map"):
        assert float(logs_all[2][f"gnorm/bev_bevkd_{n}"]) == pytest.approx(
            lams[2][n] * float(logs_all[2][f"bevkd/{n}/g_kd_unit"]), rel=1e-4)
    sd = json.loads(json.dumps(ck.state_dict()))
    assert set(sd["bev_ctrl"]) == {"det", "map"}
    ck2 = O.CKE2E2(c)
    ck2.load_state_dict(sd)
    assert ck2.bev_ctrl.state_dict() == ck.bev_ctrl.state_dict() and ck2.bev_ctrl.weights(5) == ck.bev_ctrl.weights(5)
    with pytest.raises(ValueError, match="teachers"):                   # a resumed run must keep its teachers
        O.CKE2E2(_arm_cfg(tmp_path, teachers=["map"])).load_state_dict(sd)


def test_controller_state_through_lightning_checkpoint_functions(tmp_path):
    """The CKE2E2 callback state (incl. every teacher's controller) goes through Lightning's own checkpoint hooks:
    _call_callbacks_state_dict (what dump_checkpoint stores; rank 0 writes the file) and
    _call_callbacks_load_state_dict (what every rank runs on resume)."""
    from types import SimpleNamespace

    from pytorch_lightning.trainer import call as plcall
    c = _arm_cfg(tmp_path, teachers=["det", "map"], start_mb=0)
    ck = O.CKE2E2(c)
    ck.bev_ctrl.update({"det": 2.0, "map": 0.5}, 1.0)
    ck.bev_ctrl.update({"det": 1.0, "map": float("nan")}, 1.0)
    ck.mb = 77
    agent = SimpleNamespace(_ck_e2e2=ck)
    ckpt = {"callbacks": plcall._call_callbacks_state_dict(SimpleNamespace(callbacks=[O.make_ck2_callback(agent)]))}
    ckpt = torch.load(_save(ckpt, tmp_path / "x.ckpt"), weights_only=False)
    assert ckpt["callbacks"]["CKE2E2Callback"]["bev_ctrl"] == ck.bev_ctrl.state_dict()
    fresh = O.CKE2E2(c)
    plcall._call_callbacks_load_state_dict(
        SimpleNamespace(callbacks=[O.make_ck2_callback(SimpleNamespace(_ck_e2e2=fresh))]), ckpt)
    assert fresh.bev_ctrl.state_dict() == ck.bev_ctrl.state_dict() and fresh.mb == 77
    assert fresh.bev_ctrl.weights(77) == ck.bev_ctrl.weights(77) and fresh.bev_ctrl["map"].n_skip == 1


def _save(obj, path):
    torch.save(obj, path)
    return path


# ----------------------------------------------------------------------------------------------- gloo DDP
def _ddp_worker(rank, world, port, tmp, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        f, t = TL.fresh(2, rows=[3, 7])
        f = {k: v[rank:rank + 1] for k, v in f.items()}
        t = {k: v[rank:rank + 1] for k, v in t.items()}
        if rank == 1:
            t["kd_ok_1"][:] = False          # no MAP BEV on rank 1: no MAP graph there, still in the collective
        res = {}
        for epoch in (0, 5):
            c = TL.cfg2(tmp, lat_kd={"start_mb": 0}, bev_kd={"enabled": True, "start_mb": 0,
                                                             "teachers": ["det", "map"]},
                        io_dir=f"{tmp}/io")
            ck = O.CKE2E2(c)
            ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"])
            ck.epoch, ck.epoch_frac, ck.rank, ck.world_size = epoch, epoch + 0.5, rank, world

            class Wrap(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    torch.manual_seed(0)
                    self.bev = torch.nn.Parameter(0.1 * torch.randn(1, 5000, 256))
                    self.ck_student = O.build_student_ck2(c)
                    self.ck_bev_kd = A.build_bev_kd_arm(c)
                    with torch.no_grad():               # non-zero adapters so the unit BEV gradients are measurable
                        self.ck_bev_kd.kd.adapters["det"].proj.weight.add_(0.05 * torch.randn(256, 256))
                        self.ck_bev_kd.kd.adapters["map"].proj.weight.add_(0.1 * torch.randn(256, 256))

                def forward(self, f, t, v2_logs):
                    pred = TL.fake_predictions(1, seed=rank, bev=self.bev * 1.0)
                    v2 = (pred["bev_embed"] ** 2).mean()
                    loss, logs = ck.loss(self.ck_student, f, t, pred, v2_loss=v2, v2_logs=v2_logs,
                                         bev_kd=self.ck_bev_kd)
                    return v2 + loss, logs

            model = DDP(Wrap(), find_unused_parameters=False)
            out = {}
            for mb, v2l in enumerate(({"gnorm/plan": torch.tensor(0.7)}, None)):
                model.zero_grad(set_to_none=True)
                loss, logs = model(f, t, v2l)
                loss.backward()
                ps = [p for p in model.module.parameters() if p.requires_grad]
                missing = [n for n, p in model.module.named_parameters() if p.requires_grad and p.grad is None]
                g = torch.cat([p.grad.reshape(-1) for p in ps if p.grad is not None])
                gl = [torch.zeros_like(g) for _ in range(world)]
                dist.all_gather(gl, g)
                out[mb] = {"finite": bool(torch.isfinite(g).all()), "same": bool(torch.equal(gl[0], gl[1])),
                           "missing": missing, "phase": float(logs["ck2/phase"]),
                           "lam": {n: float(logs[f"bevkd/{n}/lam"]) for n in ("det", "map")},
                           "g_unit": {n: float(logs[f"bevkd/{n}/g_kd_unit"]) for n in ("det", "map")
                                      if f"bevkd/{n}/g_kd_unit" in logs},
                           "ok_frac_map": float(logs["bevkd/map/ok_frac"]),
                           "ctrl": json.loads(json.dumps(ck.bev_ctrl.state_dict()))}
            # resume: the rank-0 state (what Lightning writes) restores this rank's continuous state exactly
            box = [ck.state_dict() if rank == 0 else None]
            dist.broadcast_object_list(box, src=0)
            ck_r = O.CKE2E2(c)
            ck_r.load_state_dict(box[0])
            out["resume_same"] = (ck_r.bev_ctrl.state_dict() == ck.bev_ctrl.state_dict()
                                  and ck_r.bev_ctrl.weights(ck.mb) == ck.bev_ctrl.weights(ck.mb))
            res[epoch] = out
        out_q.put((rank, res))
    except Exception as e:  # pragma: no cover
        import traceback
        out_q.put((rank, {"error": traceback.format_exc() + repr(e)}))
    finally:
        dist.destroy_process_group()


def test_ddp_gloo_two_process_identical_lambda(tmp_path):
    import socket

    import torch.multiprocessing as mp
    TL.batch()
    TL.make_gen(str(tmp_path / "io"), 4, [3], 16)          # row 3 (rank 0) labelled, row 7 (rank 1) -> fallback
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_ddp_worker, args=(r, 2, port, str(tmp_path), q)) for r in range(2)]
    for p in ps:
        p.start()
    res = dict(q.get(timeout=1500) for _ in ps)
    for p in ps:
        p.join(timeout=120)
    for r in (0, 1):
        assert "error" not in res[r], res[r].get("error")
    for ep, phase in ((0, 0.0), (5, 2.0)):
        for mb in (0, 1):
            a, b = res[0][ep][mb], res[1][ep][mb]
            assert a["finite"] and b["finite"] and a["same"] and b["same"], (ep, mb)
            assert not a["missing"] and not b["missing"], (a["missing"], b["missing"])
            assert a["phase"] == phase and a["ctrl"] == b["ctrl"] and a["lam"] == b["lam"]
            assert a["ok_frac_map"] == 1.0 and b["ok_frac_map"] == 0.0
        g0 = res[0][ep][0]["g_unit"]
        assert g0 == res[1][ep][0]["g_unit"] and g0["det"] > 0 and g0["map"] > 0     # all-reduced: same on both ranks
        assert set(res[0][ep][0]["ctrl"]) == {"det", "map"}
        for n in ("det", "map"):                                                      # measured on mb 0, used on mb 1
            assert res[0][ep][0]["lam"][n] == 0.0 and res[0][ep][1]["lam"][n] > 0
        assert res[0][ep][1]["lam"]["det"] != res[0][ep][1]["lam"]["map"]           # one controller per teacher
        assert res[0][ep]["resume_same"] and res[1][ep]["resume_same"]
