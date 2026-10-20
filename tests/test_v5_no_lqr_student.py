"""refiner v5 no-LQR student (user 2026-10-20: the student drops the LQR rollout, the teachers keep it): exec_model 'ref'
= the tracker's 41-state reference without the tracker (lqr.reference_rollout), wired through candidate_states (the
range readout / relation re-reads), refine_loss (L_refine's X0 / X1) and V5Cfg; the teachers' two rollouts of the same
candidates shared inside range_readout.shared_states().  'lqr' paths unchanged.  CPU only, seconds:

  cd /workspace/yongjae/SSR-v5 && env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
      /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v5_no_lqr_student.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck.v4 import lqr as LQR  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import range_readout as RR  # noqa: E402


def _traj(B: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(1, 9, dtype=torch.float32)[None] * 0.5
    v = 2 + 10 * torch.rand(B, 1, generator=g)
    x = v * t + 0.2 * torch.randn(B, 8, generator=g).cumsum(1)
    y = 0.3 * torch.randn(B, 8, generator=g).cumsum(1)
    h = 0.05 * torch.randn(B, 8, generator=g).cumsum(1)
    return torch.stack([x, y, h], -1)


def test_reference_rollout_is_the_trackers_reference_with_kinematic_columns():
    tr = _traj(5)
    ego = torch.tensor([[7.0, 0.3], [0.0, 0.0], [12.0, -1.0], [3.0, 0.5], [9.0, 0.0]])
    st = LQR.reference_rollout(tr, ego)
    assert st.shape == (5, LQR.NT, 6) and st.dtype == torch.float32
    ref = LQR.reference_states(tr)
    assert torch.equal(st[..., :2], ref[..., :2])                               # positions = the reference
    assert torch.allclose(st[..., 2], ref[..., 2], atol=1e-5)                   # drivable curvature: its heading too
    assert torch.equal(st[:, 0, 3], ego[:, 0]) and torch.equal(st[:, 0, 4], ego[:, 1])
    assert torch.equal(st[:, 0, :3], torch.zeros(5, 3))
    d = ref[:, 1:, :2] - ref[:, :-1, :2]
    vs = (d[..., 0] * torch.cos(st[:, :-1, 2]) + d[..., 1] * torch.sin(st[:, :-1, 2])) / LQR.DT
    assert torch.allclose(st[:, 1:40, 3], vs[:, 1:], atol=1e-4)                 # v_n moves state n (n = 1..39)
    assert torch.equal(st[:, 40, 3], st[:, 39, 3])
    assert torch.allclose(st[:, 1:, 4], (st[:, 1:, 3] - st[:, :-1, 3]) / LQR.DT)
    assert torch.equal(st[:, 1:, 5], torch.zeros(5, LQR.NT - 1))


def test_reference_rollout_keeps_a_near_stop_plan_pointing_forward():
    """v2 stop parents: the plan creeps ~0.6 m while its knot headings flip by ~2.9 rad; the bicycle cannot turn
    faster than len tan(pi / 3) / L, so the 'ref' heading stays forward (the plain reference points backwards)."""
    x = torch.tensor([0.45, 0.55, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57])
    h = torch.tensor([-0.02, -2.89, -2.82, -2.85, -2.85, -2.85, -2.85, -2.85])
    tr = torch.stack([x, torch.zeros(8), h], -1)[None]
    st = LQR.reference_rollout(tr, torch.tensor([[1.62, 0.0]]))
    ref = LQR.reference_states(tr)
    assert float(ref[0, 15, 2].abs()) > 2.5                                      # the reference itself flips
    lens = torch.linalg.vector_norm(ref[0, 1:, :2] - ref[0, :-1, :2], dim=-1)
    bound = float(lens.sum()) * math.tan(LQR.MAX_STEER) / LQR.WHEEL_BASE
    assert float(st[0, :, 2].abs().max()) <= bound + 1e-5 < 0.5
    assert bool((st[0, 1:40, 3] >= -1e-6).all())                                 # it creeps forward, not backwards


def test_reference_rollout_constant_speed_straight_line():
    t = torch.arange(1, 9, dtype=torch.float64)[None] * 0.5
    tr = torch.stack([8.0 * t, torch.zeros_like(t), torch.zeros_like(t)], -1)          # 8 m/s straight
    st = LQR.reference_rollout(tr, torch.tensor([[8.0, 0.0]], dtype=torch.float64))
    assert st.dtype == torch.float64
    assert torch.allclose(st[0, :, 3], torch.full((LQR.NT,), 8.0, dtype=torch.float64))
    assert torch.allclose(st[0, :, 4], torch.zeros(LQR.NT, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(st[0, :, 0], torch.arange(LQR.NT, dtype=torch.float64) * 0.8)
    assert torch.allclose(st[0, :, 2], torch.zeros(LQR.NT, dtype=torch.float64))


def test_reference_rollout_broadcasts_like_simulate_and_has_finite_gradients_at_standstill():
    tr = _traj(6).reshape(2, 3, 8, 3)
    ego = torch.tensor([[[5.0, 0.0]], [[0.0, 0.0]]])                             # [T, 1, 2] as refine_loss passes it
    st = LQR.reference_rollout(tr, ego)
    assert st.shape == (2, 3, LQR.NT, 6)
    assert torch.allclose(st[1, :, 0, 3], torch.zeros(3))
    assert LQR.simulate(tr, ego).shape == st.shape
    z = torch.zeros(2, 8, 3, requires_grad=True)                                 # standing still: no norm, no NaN
    LQR.reference_rollout(z, torch.zeros(2, 2)).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_executor_dispatch():
    assert LQR.executor("lqr") is LQR.simulate
    assert LQR.executor("ref") is LQR.reference_rollout
    with pytest.raises(ValueError):
        LQR.executor("plan")


def test_candidate_states_ref_and_lqr_paths():
    tr = _traj(4)
    tr[2, 3, 0] = float("nan")                                                   # non-finite rows -> zeros (both)
    vx, ax = torch.tensor([6.0, float("nan"), 2.0, 9.0]), torch.tensor([0.1, 0.0, 0.0, -0.4])
    c = torch.where(torch.isfinite(tr).flatten(1).all(1)[:, None, None], tr, torch.zeros_like(tr))
    e = torch.stack([torch.nan_to_num(vx, nan=0.0), ax], -1)
    st, t = RR.candidate_states(tr, vx, ax, "ref")
    assert torch.equal(st, LQR.reference_rollout(c, e, dtype=torch.float32)) and t.shape == (LQR.NT,)
    st_l, t_l = RR.candidate_states(tr, vx, ax, "lqr")
    assert torch.equal(st_l, LQR.simulate(c, e, dtype=torch.float32)) and torch.equal(t, t_l)
    st_p, t_p = RR.candidate_states(tr, vx, ax, "plan")
    assert st_p.shape == (4, 9, 6) and t_p.shape == (9,)
    with pytest.raises(ValueError):
        RR.candidate_states(tr, vx, ax, "bicycle")


def test_shared_states_rolls_equal_inputs_out_once(monkeypatch):
    calls = []
    real = LQR.simulate

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    refs = []
    real_ref = LQR.reference_rollout

    def counting_ref(*a, **k):
        refs.append(1)
        return real_ref(*a, **k)

    monkeypatch.setattr(LQR, "simulate", counting)
    monkeypatch.setattr(LQR, "reference_rollout", counting_ref)
    tr, vx, ax = _traj(3), torch.tensor([5.0, 6.0, 7.0]), torch.zeros(3)
    s1, _ = RR.candidate_states(tr, vx, ax, "lqr")
    s2, _ = RR.candidate_states(tr, vx, ax, "lqr")
    assert len(calls) == 2                                                        # no scope: no reuse
    calls.clear()
    with RR.shared_states():
        a1, _ = RR.candidate_states(tr, vx, ax, "lqr")
        a2, _ = RR.candidate_states(tr.clone(), vx.clone(), ax.clone(), "lqr")   # equal values, new tensors
        assert len(calls) == 1 and torch.equal(a1, a2) and a1 is not a2
        RR.candidate_states(tr, vx + 1.0, ax, "lqr")                              # different ego: rolled out
        RR.candidate_states(tr[:2], vx[:2], ax[:2], "lqr")                         # different shape: rolled out
        assert len(calls) == 3
        r1, _ = RR.candidate_states(tr, vx, ax, "ref")                             # other model: not the LQR entry
        assert len(calls) == 3 and len(refs) == 1 and not torch.equal(r1, a1)
        r2, _ = RR.candidate_states(tr, vx, ax, "ref")                             # ... but its own entry is reused
        assert len(refs) == 1 and torch.equal(r1, r2)
        a1.add_(1.0)                                                               # a consumer writing in place ...
        a3, _ = RR.candidate_states(tr, vx, ax, "lqr")
        assert torch.equal(a3, s1)                                                 # ... never reaches the next one
    RR.candidate_states(tr, vx, ax, "lqr")
    assert len(calls) == 4 and RR._MEMO is None                                   # scope closed


def test_v5cfg_accepts_ref_and_lqr_only():
    from navsim.agents.para_ssr.ck.v4.config import V4Cfg
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    for m in ("lqr", "ref"):
        V5Cfg(unknown_ev=True).validate(V4Cfg.range_arm(exec_model=m))
    with pytest.raises(ValueError):
        V5Cfg(unknown_ev=True).validate(V4Cfg.range_arm(exec_model="plan"))


def _gt_one_car():
    from tests.test_v5_review_fixes import _bicycle_gt
    from navsim.agents.para_ssr.refiner import gt_future as GF
    gt = _bicycle_gt()
    gt["kf"][0, 0, :, GF.KF_X] = 30.0
    gt["kf"][0, 0, :, GF.KF_Y] = 0.0
    gt["kf"][0, 0, :, GF.KF_H] = 0.0
    gt["kf"][0, 0, :, GF.KF_VY] = 0.0
    gt["first"] = torch.tensor([[[4.5, 1.9, 0.0, 0.0, 0.0, 0.0]]])
    gt["meta"][0, 0, 0] = 0
    gt["sdf"] = torch.full((1, 320, 256), 5.0)
    return gt


def _straight(v: float) -> torch.Tensor:
    t = torch.arange(1, 9, dtype=torch.float32)[None] * 0.5
    return torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)


def test_refine_loss_ref_never_calls_the_tracker_and_keeps_the_braking_preference(monkeypatch):
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL
    gt, w = _gt_one_car(), RL.parse_weights("")
    fast, slow = _straight(10.0)[None], _straight(3.0)[None]
    v0, a0 = torch.tensor([10.0]), torch.tensor([0.0])
    l_lqr = float(RL.refine_loss(slow, fast, v0, a0, gt, w)[0])                  # default: the tracker (teachers)

    def boom(*a, **k):
        raise AssertionError("the 'ref' path must not run the LQR rollout")

    monkeypatch.setattr(LQR, "simulate", boom)
    tau1 = slow.clone().requires_grad_(True)
    l_slow, st = RL.refine_loss(tau1, fast, v0, a0, gt, w, exec_model="ref")
    l_fast, _ = RL.refine_loss(fast, slow, v0, a0, gt, w, exec_model="ref")
    assert float(l_slow.detach()) < 0 < float(l_fast)                            # braking before the stopped car wins
    l_slow.backward()
    assert torch.isfinite(tau1.grad).all() and float(tau1.grad.abs().sum()) > 0
    assert math.isfinite(l_lqr) and l_lqr < 0
    with pytest.raises(AssertionError):
        RL.refine_loss(slow, fast, v0, a0, gt, w)                                # 'lqr' still the tracker


def test_v5_arm_set_student_reads_ref():
    lines = [l.strip() for l in (REPO / "tools/ck/e2e2/v5_arm.set").read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    assert "v4.exec_model=ref" in lines and "v4.exec_model=lqr" not in lines


def test_refine_loss_ref_drops_the_comfort_factor():
    """'ref' accelerations are finite differences of the plan: the comfort factor needs the tracker, so 'ref' runs
    L_refine with cmf = 0 whatever the weights say ('lqr' keeps it)."""
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL
    gt, w = _gt_one_car(), RL.parse_weights("")
    assert w["cmf"] > 0
    tau0 = _straight(10.0)[None]
    jerky = tau0.clone()
    jerky[..., 1:, 1] += torch.tensor([0.0, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5])   # lateral zig-zag: comfort fails
    v0, a0 = torch.tensor([10.0]), torch.tensor([0.0])
    l_ref, st_ref = RL.refine_loss(jerky, tau0, v0, a0, gt, w, exec_model="ref")
    l_ref0, _ = RL.refine_loss(jerky, tau0, v0, a0, gt, dict(w, cmf=0.0), exec_model="ref")
    assert float(l_ref) == float(l_ref0) and st_ref["ref_p_c1"] == 1.0
    _, st_lqr = RL.refine_loss(jerky, tau0, v0, a0, gt, w)
    assert st_lqr["ref_p_c1"] < 1.0


def test_teachers_read_zkd_targets_at_the_students_states():
    """Z-KD with a no-LQR student: score_set(zq_exec='ref') of an 'lqr' net adds haz['zq_kd'] = the reads of its own
    fields at the 'ref' states (== the same weights' zq with exec 'ref', bit for bit), every score unchanged;
    teacher_forward(extras, zq_exec) hands that to Z-KD."""
    import copy
    import dataclasses
    from tests.test_v5_student_completion import _pool_inputs, _student_net
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows, teacher_forward
    net, cfg = _student_net()
    net.eval()
    net2 = copy.deepcopy(net)
    net2.v4 = dataclasses.replace(net.v4, exec_model="ref")
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        rel = student_rel_rows(det, cfg.v5)
        o0 = net.score_set(feat, mem, ego, cand, par, rel_objs=rel)
        o1 = net.score_set(feat, mem, ego, cand, par, rel_objs=rel, zq_exec="ref")
        o2 = net2.score_set(feat, mem, ego, cand, par, rel_objs=rel)
        o3 = net.score_set(feat, mem, ego, cand, par, rel_objs=rel, zq_exec="lqr")       # own model: nothing added
    assert "zq_kd" not in o0["haz"] and "zq_kd" not in o3["haz"]
    assert torch.equal(o1["score_logit"], o0["score_logit"]) and torch.equal(o1["haz"]["zq"], o0["haz"]["zq"])
    assert torch.equal(o1["haz"]["zq_kd"], o2["haz"]["zq"])
    assert not torch.equal(o1["haz"]["zq_kd"], o0["haz"]["zq"])
    ex, ex0 = {}, {}
    with torch.no_grad():
        s1 = teacher_forward(net, bev, cand, ego, None, par, rel_objs=rel, extras=ex, zq_exec="ref")
        s0 = teacher_forward(net, bev, cand, ego, None, par, rel_objs=rel, extras=ex0)
    assert torch.equal(ex["zq"], o2["haz"]["zq"].float()) and torch.equal(ex0["zq"], o0["haz"]["zq"].float())
    assert torch.equal(s1[0], s0[0])


def test_v5_terms_of_a_ref_student_never_run_the_tracker(monkeypatch):
    """CKE2E2._v5_terms (L_refine on, corrector on) of a 'ref' student: refine_loss gets the student's exec model, so
    the tracker never runs and the 'ref' comfort-fail stat is not logged; the same net on 'lqr' does run it."""
    import dataclasses
    from tests.test_v5_student_completion import _student_net, _zkd_terms
    net, cfg0 = _student_net(rel_modes="cv+fut", top_feats="v2")
    q = dict(cfg0.v5, lambda_ref=1.0, lambda_rel=0.5, lambda_rel_cv=0.25, rel_det_grad=True,
             rel_det_grad_matched=True)
    calls = []
    real = LQR.simulate

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(LQR, "simulate", counting)
    _zkd_terms(net, cfg0, q)
    assert calls                                                                  # 'lqr' student: the tracker runs
    calls.clear()
    net.v4 = dataclasses.replace(net.v4, exec_model="ref")
    inner, logs, *_ = _zkd_terms(net, cfg0, q)
    assert not calls and math.isfinite(float(inner))
    assert "v5/ref_loss" in logs and not any("cmf_fail" in k for k in logs)


def test_the_kd_loss_hands_the_students_exec_model_to_the_teachers():
    """CKE2E2.loss -> TeacherPair2.run(zq, zq_exec=student exec model) -> teacher_forward(zq_exec) (the end-to-end
    path runs in tools/ck/v5/smoke_student_v5.py; here the two hand-overs are pinned in the source)."""
    import inspect
    from navsim.agents.para_ssr.ck import online2 as O2
    src = inspect.getsource(O2.CKE2E2.loss)
    assert 'tkw["zq_exec"] = student.v4.exec_model' in src
    run_src = inspect.getsource(O2.TeacherPair2.run)
    assert '"zq_exec": zq_exec' in run_src and "shared_states()" in run_src


def test_z_stats_records_must_match_the_students_exec_model():
    from navsim.agents.para_ssr.ck.v5 import zadapt as ZA
    from tests.test_v5_student_completion import _zrec
    old = _zrec("T")                                                              # no zq_exec: the teacher's LQR
    ZA.check_z_stats(old, "T", exec_model="lqr")
    with pytest.raises(ValueError, match="zq-exec ref"):
        ZA.check_z_stats(old, "T", exec_model="ref")
    ZA.check_z_stats(_zrec("T", zq_exec="ref"), "T", exec_model="ref")
    with pytest.raises(ValueError):
        ZA.check_z_stats(_zrec("T", zq_exec="ref"), "T", exec_model="lqr")
    ZA.check_z_stats(old, "T")                                                    # no student given: not checked
    from tools.ck.v5 import fit_z_stats as FZ
    r = Path("/x/v5T_rel6")
    assert FZ.default_out(r, "last").name == "v5T_rel6__last.json"
    assert FZ.default_out(r, "last", "ref").name == "v5T_rel6__last__zqref.json"


_RUNS = Path("/home/external-user/ssd/yongjae_refiner/ck/v5/train")


@pytest.mark.skipif(not ((_RUNS / "v5T_rel5/ckpt_last.pt").is_file() and (_RUNS / "v5M_unk5/ckpt_last.pt").is_file()),
                    reason="needs the v5 teacher runs v5T_rel5 / v5M_unk5 and the navtrain_train packed cache")
def test_teacher_pair_reads_zkd_targets_at_the_students_states_with_the_real_teachers():
    """TeacherPair2.run(zq=True, zq_exec='ref') on the real v5 teachers (CPU, 2 tokens x 48 columns): zq_det / zq_map
    = each teacher's zq on a 'ref'-configured copy, the scores bit-identical to zq_exec 'lqr' / None."""
    import copy
    import dataclasses
    import numpy as np
    from navsim.agents.para_ssr.ck import cands2 as C2
    from navsim.agents.para_ssr.ck.online2 import TeacherPair2, _d_variants, infer_pool, teacher_forward
    from navsim.agents.para_ssr.ck.temporal.config import OBJ_INPUT_KEYS
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    from tools.ck.e2e2 import fit_kd_calib as FK
    tp = TeacherPair2(str(_RUNS / "v5T_rel5"), str(_RUNS / "v5M_unk5"), "last", device="cpu", amp=False)
    rows = np.array([1000, 1997], dtype=np.int64)
    iT = FK.run_info(_RUNS / "v5T_rel5", "last")
    dsT = CKDataset("navtrain_train", bev="T", k=16, labels=None, gt=False, rows=rows,
                    objects=FK.objects_spec(iT["v5_rel"]))
    dsM = CKDataset("navtrain_train", bev="M", k=16, labels=None, gt=False, rows=rows)
    bT, bM = collate_ck([dsT[i] for i in range(2)]), collate_ck([dsM[i] for i in range(2)])
    cand16, status = bT["cand"].float(), bT["status"].float()
    V = C2.online_variants(cand16, C2.v0_from_status(status), _d_variants())
    cand96, _, par96 = infer_pool(V["traj96"], cand16)
    cand, par = cand96[:, :48].contiguous(), par96[:, :48].contiguous()
    rel = {k: bT[k] for k in OBJ_INPUT_KEYS}
    kw = dict(parents=par, rel_objs_det=rel, rel_ok_det=bT["obj_ok"].bool(), zq=True)
    o_ref = tp.run(bT["bev"], bT["bev_ok"], bM["bev"], bM["bev_ok"], cand, status, zq_exec="ref", **kw)
    o_lqr = tp.run(bT["bev"], bT["bev_ok"], bM["bev"], bM["bev_ok"], cand, status, zq_exec="lqr", **kw)
    o_non = tp.run(bT["bev"], bT["bev_ok"], bM["bev"], bM["bev_ok"], cand, status, **kw)
    for k, v in o_lqr.items():
        if torch.is_tensor(v) and not k.startswith("zq_"):
            assert torch.equal(o_ref[k], v), k                                   # scores etc. untouched
    assert torch.equal(o_lqr["zq_det"], o_non["zq_det"]) and torch.equal(o_lqr["zq_map"], o_non["zq_map"])
    ego = ego_inputs(status)
    for name, net, bev, r in (("det", tp.det, bT["bev"], rel), ("map", tp.map, bM["bev"], None)):
        n2 = copy.deepcopy(net)
        n2.v4 = dataclasses.replace(net.v4, exec_model="ref")
        ex = {}
        with torch.no_grad():
            teacher_forward(n2, bev.float(), cand, ego, None, par, **({"rel_objs": r} if r is not None else {}),
                            extras=ex)
        assert torch.equal(o_ref[f"zq_{name}"], ex["zq"]), name
        assert not torch.equal(o_ref[f"zq_{name}"], o_lqr[f"zq_{name}"]), name
