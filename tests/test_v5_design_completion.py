"""refiner v5 design completion (design refiner_v5_design_final.md, SSR/report/97 + 101 §6), group 'det': the student
det head's V1b (velocity rows only on moving matched GT), V1c (stop_branch: the deployed velocity v = (1 - sigmoid(stop))
v_hat inside the head), the ACC aux head (a_lon / p_stop2s on det_hidden + GT keyframe-speed targets of the target
builder), p_stop_i in the student reliability MLP (V5Cfg.rel_pstop), and the Stage-2 arm files.

Default off = the head / loss / target builder of HEAD 1abf369 bit for bit (exact constants of a seeded tiny head, the
target cache name); each option's intended effect on constructed cases (which rows the losses read, where gradients
go, target rules); DDP safety (every new parameter gets a gradient on a batch without GT).  CPU only, seconds:

  cd /workspace/yongjae/SSR-v5 && env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
      /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v5_design_completion.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.modules.det_motion_head import ParaDetMotionHead  # noqa: E402
from navsim.agents.para_ssr.modules.losses import hungarian_assign_det, normalize_bbox  # noqa: E402

ON = dict(vel_moving_only=True, stop_branch=True, acc_head=True)


# ----------------------------------------------------------------------------------------------- det head helpers
def _head(**kw) -> ParaDetMotionHead:
    """the seeded tiny head of the HEAD 1abf369 reference values (2 decoder layers, 6 queries, embed 16)."""
    torch.manual_seed(123)
    return ParaDetMotionHead(num_query=6, num_classes=7, embed_dims=16, bev_h=2, bev_w=4, code_size=10, fut_ts=2,
                             fut_mode=3, num_decoder_layers=2, num_heads=2, feedforward_channels=32, **kw)


def _bev() -> torch.Tensor:
    return torch.randn(2, 8, 16, generator=torch.Generator().manual_seed(7))


def _gt():
    """batch 0: moving vehicle 3 m/s, static vehicle 0.2 m/s, cone with an annotated 1 m/s, pedestrian 0.7 m/s;
    batch 1: no GT.  Physical rows (x_right, y_fwd, z, w, l, h, yaw, vx, vy) + class."""
    gt = torch.zeros(2, 5, 9)
    lab = torch.zeros(2, 5, dtype=torch.long)
    val = torch.zeros(2, 5, dtype=torch.bool)
    rows = [(-3.0, 8.0, 0.0, 1.9, 4.5, 1.6, 0.1, 0.5, 3.0, 0),
            (4.0, 12.0, 0.0, 2.0, 4.8, 1.7, -0.2, 0.1, 0.17, 0),
            (1.0, 5.0, 0.0, 0.4, 0.4, 0.8, 0.0, 0.6, 0.8, 3),
            (-6.0, 20.0, 0.0, 0.7, 0.7, 1.8, 1.0, 0.7, 0.0, 1)]
    for i, r in enumerate(rows):
        gt[0, i] = torch.tensor(r[:9])
        lab[0, i] = r[9]
        val[0, i] = True
    fut = torch.randn(2, 5, 2, 2, generator=torch.Generator().manual_seed(5))
    return gt, lab, val, fut, torch.ones(2, 5, 2)


def _kf(shape=(2, 5)):
    """keyframe speeds / presence for _gt(): decelerating mover, the rest constant, every keyframe annotated."""
    sp = torch.zeros(*shape, 5)
    sp[0, 0] = torch.tensor([3.0, 2.5, 1.0, 0.3, 0.0])
    sp[0, 1] = 0.2
    sp[0, 3] = 0.7
    return sp, torch.ones(*shape, 5, dtype=torch.bool)


def _train_out(h: ParaDetMotionHead):
    h.train()
    torch.manual_seed(11)
    return h(_bev(), return_hidden=True)


# loss values of _head() / _head(vel_code_weights=(1.05, 0.41)) at HEAD 1abf369 (scratch dump, same seeds / inputs)
HEAD_LOSS_OFF = {"loss_cls_d0": 2.2362828254699707, "loss_bbox_d0": 3.9271228313446045, "loss_cls": 2.231201410293579,
                 "loss_bbox": 3.862274169921875, "loss_traj": 0.7121931910514832,
                 "loss_traj_cls": 0.22348351776599884}
HEAD_LOSS_V1 = dict(HEAD_LOSS_OFF, loss_bbox_d0=4.032233238220215, loss_bbox=3.977869987487793)
HEAD_TARGET_NAME = "para_ssr_target_0edd648f"      # ParaSSRTargetBuilder(ParaSSRConfig(), 4 s / 0.5 s) at 1abf369
OUT_KEYS = {"all_cls_scores", "all_bbox_preds", "traj_preds", "traj_cls_preds", "det_hidden", "motion_hidden"}


def _synthetic_preds(h: ParaDetMotionHead, gt, lab, n_layers: int = 1, seed: int = 0):
    """predictions whose query i sits exactly on GT i of batch 0 with a confident class logit (Hungarian = identity),
    + every output key the head's options add (raw velocity / stop logits / acc)."""
    g = torch.Generator().manual_seed(seed)
    B, Q = gt.shape[0], h.num_query
    cls = torch.full((n_layers, B, Q, 7), -5.0)
    code = torch.randn(n_layers, B, Q, 10, generator=g) * 0.1
    n = min(Q, gt.shape[1])
    code[:, 0, :n] = code[:, 0, :n] + normalize_bbox(gt[0, :n])
    for i in range(n):
        cls[:, 0, i, int(lab[0, i])] = 5.0
    p = {"all_cls_scores": cls, "all_bbox_preds": code,
         "traj_preds": torch.randn(B, Q, h.fut_mode, h.fut_ts, 2, generator=g),
         "traj_cls_preds": torch.randn(B, Q, h.fut_mode, generator=g)}
    if h.stop_branches is not None:
        p["all_vel_raw"] = torch.randn(n_layers, B, Q, 2, generator=g)
        p["all_stop_logits"] = torch.randn(n_layers, B, Q, generator=g)
    if h.acc_head is not None:
        p["acc_preds"] = torch.randn(B, Q, generator=g)
        p["stop2s_logits"] = torch.randn(B, Q, generator=g)
    return p


def _small(Q: int, **kw) -> ParaDetMotionHead:
    return ParaDetMotionHead(num_query=Q, num_classes=7, embed_dims=16, bev_h=2, bev_w=4, code_size=10, fut_ts=2,
                             fut_mode=3, num_decoder_layers=1, num_heads=2, feedforward_channels=32, **kw)


def _focal(logit: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """sigmoid focal (alpha 0.25, gamma 2) of a 1-class logit, per row (pos = the class-0 target)."""
    p = torch.sigmoid(logit)
    t = pos.to(logit.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, t, reduction="none")
    pt = (1 - p) * t + p * (1 - t)
    return bce * (0.25 * t + 0.75 * (1 - t)) * pt.pow(2)


# ----------------------------------------------------------------------------------------------- default off
def test_det_off_is_head_1abf369_bit_for_bit():
    """no option -> no new module / key, the exact HEAD 1abf369 losses (plain and V1); the old weighted L1 by hand."""
    for kw, want in (({}, HEAD_LOSS_OFF), ({"vel_code_weights": (1.05, 0.41)}, HEAD_LOSS_V1)):
        h = _head(**kw)
        assert h.stop_branches is None and h.acc_head is None and not h.vel_moving_only
        assert not any(k.startswith(("stop_branches", "acc_head")) for k in h.state_dict())
        o = _train_out(h)
        assert set(o) == OUT_KEYS
        L = h.loss(o, *_gt())
        assert {k: v.item() for k, v in L.items()} == want
    # independent re-implementation of the box L1 for a synthetic, identity-matched prediction
    h = _small(4)
    gt, lab, val, _, _ = _gt()
    p = _synthetic_preds(h, gt, lab)
    L = h.loss(p, gt, lab, val)
    cw = torch.tensor([1.0] * 8 + [0.2, 0.2])
    ref = ((p["all_bbox_preds"][0, 0, :4] - normalize_bbox(gt[0, :4])).abs() * cw).sum() / 4 * 0.25
    assert torch.allclose(L["loss_bbox"], ref, rtol=1e-6, atol=0)


def test_det_options_keep_existing_init_and_outputs():
    """the options' modules are built on a forked RNG stream: every existing parameter equals the off head's, the class /
    box (dims 0..7) / motion outputs are unchanged; the deployed velocity is exactly (1 - sigmoid(stop)) x the raw one,
    the raw one is the off head's velocity, and the gate starts open (sigmoid ~ 0.01)."""
    off, on = _head(), _head(**ON)
    sd_off, sd_on = off.state_dict(), on.state_dict()
    assert list(sd_on)[: len(sd_off)] == list(sd_off)
    assert all(torch.equal(sd_off[k], sd_on[k]) for k in sd_off)
    new = sorted(set(sd_on) - set(sd_off))
    assert new == sorted([f"stop_branches.{i}.{w}" for i in range(2) for w in ("weight", "bias")]
                         + [k for k in sd_on if k.startswith("acc_head.")])
    off.eval(), on.eval()
    with torch.no_grad():
        a, b = off(_bev(), return_hidden=True), on(_bev(), return_hidden=True)
    assert set(b) == OUT_KEYS | {"all_stop_logits", "all_vel_raw", "acc_preds", "stop2s_logits"}
    for k in OUT_KEYS - {"all_bbox_preds"}:
        assert torch.equal(a[k], b[k]), k
    assert torch.equal(a["all_bbox_preds"][..., :8], b["all_bbox_preds"][..., :8])
    assert torch.equal(b["all_vel_raw"], a["all_bbox_preds"][..., 8:10])
    gate = 1.0 - b["all_stop_logits"].sigmoid()
    assert torch.equal(b["all_bbox_preds"][..., 8:10], b["all_vel_raw"] * gate[..., None])
    assert b["all_stop_logits"].shape == (2, 2, 6) and b["acc_preds"].shape == b["stop2s_logits"].shape == (2, 6)
    assert 0.002 < float(b["all_stop_logits"].sigmoid().mean()) < 0.05


def test_det_option_refusals():
    with pytest.raises(ValueError, match="stop_branch"):
        _small(3, vel_moving_only=True)
    with pytest.raises(ValueError, match="code_size"):
        ParaDetMotionHead(num_query=3, embed_dims=16, bev_h=2, bev_w=4, code_size=8, code_weights=[1.0] * 8,
                          num_decoder_layers=1, num_heads=2, feedforward_channels=32, stop_branch=True)
    with pytest.raises(ValueError, match="stop_speed"):
        _small(3, stop_branch=True, stop_speed=0.0)


# ----------------------------------------------------------------------------------------------- V1b
def test_v1b_velocity_rows_only_on_moving_agents():
    """GT: static vehicle 0.2 m/s, cone (annotated 1 m/s), moving vehicle 3 m/s, matched to queries 0, 1, 2: the raw
    velocity of queries 0 / 1 does not enter the box L1, query 2's does; the divisor is still all 3 matches; the
    Hungarian assignment is the same with the options on and off (velocity is not in the cost)."""
    gt = torch.zeros(1, 3, 9)
    gt[0, 0] = torch.tensor([4.0, 12.0, 0.0, 2.0, 4.8, 1.7, -0.2, 0.12, 0.16, ])
    gt[0, 1] = torch.tensor([1.0, 5.0, 0.0, 0.4, 0.4, 0.8, 0.0, 0.6, 0.8])
    gt[0, 2] = torch.tensor([-3.0, 8.0, 0.0, 1.9, 4.5, 1.6, 0.1, 0.0, 3.0])
    lab = torch.tensor([[0, 3, 0]])
    val = torch.ones(1, 3, dtype=torch.bool)
    h = _small(3, vel_moving_only=True, stop_branch=True)
    p = _synthetic_preds(h, gt, lab)
    q, g = hungarian_assign_det(p["all_cls_scores"][0, 0], p["all_bbox_preds"][0, 0], normalize_bbox(gt[0]), lab[0])
    assert q.tolist() == [0, 1, 2] and g.tolist() == [0, 1, 2]
    base = h.loss(p, gt, lab, val)
    for qq, same in ((0, True), (1, True), (2, False)):
        p2 = dict(p, all_vel_raw=p["all_vel_raw"].clone())
        p2["all_vel_raw"][0, 0, qq, 0] += 1.0                      # one component (two can cancel in the L1)
        got = h.loss(p2, gt, lab, val)["loss_bbox"]
        assert (float(got) == float(base["loss_bbox"])) == same, qq
    # hand value: rows 0..7 of every match + rows 8, 9 of the mover only, / 3 matches, x 0.25
    tgt = normalize_bbox(gt[0])
    code = torch.cat([p["all_bbox_preds"][0, 0, :, :8], p["all_vel_raw"][0, 0]], -1)
    w = torch.ones(3, 10)
    w[:, 8:] = 0.2
    w[:2, 8:] = 0.0
    assert torch.allclose(base["loss_bbox"], ((code - tgt).abs() * w).sum() / 3 * 0.25, rtol=1e-6, atol=0)
    # without V1b the static rows' raw velocity counts
    h_c = _small(3, stop_branch=True)
    p3 = dict(p, all_vel_raw=p["all_vel_raw"].clone())
    p3["all_vel_raw"][0, 0, 0, 0] += 1.0
    assert float(h_c.loss(p3, gt, lab, val)["loss_bbox"]) != float(h_c.loss(p, gt, lab, val)["loss_bbox"])
    # the class / motion terms (they depend on the assignment) equal the plain head's
    plain = _small(3).loss({k: p[k] for k in ("all_cls_scores", "all_bbox_preds", "traj_preds", "traj_cls_preds")},
                           gt, lab, val)
    for k in ("loss_cls", "loss_traj", "loss_traj_cls"):
        assert float(plain[k]) == float(base[k]), k


def test_v1b_agent_config_refusals():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    cfg = ParaSSRConfig()
    ParaSSRAgent._validate_config(cfg, ts)
    ParaSSRAgent._validate_config(replace(cfg, det_vel_moving_only=True, det_stop_branch=True, det_acc_head=True), ts)
    with pytest.raises(ValueError, match="det_stop_branch"):
        ParaSSRAgent._validate_config(replace(cfg, det_vel_moving_only=True), ts)
    with pytest.raises(ValueError, match="use_det_motion_head"):
        ParaSSRAgent._validate_config(replace(cfg, use_det_motion_head=False, use_task_interaction=False,
                                              det_acc_head=True), ts)


# ----------------------------------------------------------------------------------------------- V1c
def test_v1c_stop_loss_labels_and_l1_on_raw_velocity():
    """GT {vehicle 0.3 m/s -> stop, vehicle 3 m/s -> moving, cone -> stop}: loss_stop = the hand focal / 3 matches; the
    stop logits never enter the box L1; per-layer keys; missing raw velocity is refused."""
    gt = torch.zeros(1, 3, 9)
    gt[0, 0] = torch.tensor([4.0, 12.0, 0.0, 2.0, 4.8, 1.7, -0.2, 0.0, 0.3])
    gt[0, 1] = torch.tensor([-3.0, 8.0, 0.0, 1.9, 4.5, 1.6, 0.1, 0.0, 3.0])
    gt[0, 2] = torch.tensor([1.0, 5.0, 0.0, 0.4, 0.4, 0.8, 0.0, 0.6, 0.8])
    lab = torch.tensor([[0, 0, 3]])
    val = torch.ones(1, 3, dtype=torch.bool)
    h = _small(4, stop_branch=True)
    p = _synthetic_preds(h, gt, lab, n_layers=2)
    L = h.loss(p, gt, lab, val)
    assert {"loss_stop", "loss_stop_d0"} <= set(L)
    s = p["all_stop_logits"][-1, 0, :3]
    ref = _focal(s, torch.tensor([True, False, True])).sum() / 3
    assert torch.allclose(L["loss_stop"], ref, rtol=1e-6, atol=0)
    p2 = dict(p, all_stop_logits=p["all_stop_logits"] + 3.0)
    p2["all_stop_logits"][-1, 0, 3] += 50.0                     # the unmatched query carries no stop target
    L2 = h.loss(p2, gt, lab, val)
    assert float(L2["loss_bbox"]) == float(L["loss_bbox"]) and float(L2["loss_stop"]) != float(L["loss_stop"])
    ref2 = _focal(p2["all_stop_logits"][-1, 0, :3], torch.tensor([True, False, True])).sum() / 3
    assert torch.allclose(L2["loss_stop"], ref2, rtol=1e-6, atol=0)
    with pytest.raises(KeyError, match="all_vel_raw"):
        h.loss({k: v for k, v in p.items() if k != "all_vel_raw"}, gt, lab, val)


def test_new_parameters_get_gradients_without_gt():
    """DDP safety (find_unused_parameters False): on a batch without any valid GT, and on one with GT, every parameter
    of the head with all options on receives a (finite) gradient."""
    gt, lab, val, fut, fm = _gt()
    sp, pr = _kf()
    for v in (torch.zeros_like(val), val):
        h = _head(**ON)
        o = _train_out(h)
        L = h.loss(o, gt, lab, v, fut, fm, gt_kf_speed=sp, gt_kf_present=pr)
        assert {"loss_stop", "loss_stop_d0", "loss_acc", "loss_stop2s"} <= set(L)
        assert all(torch.isfinite(x) for x in L.values())
        sum(L.values()).backward()
        miss = [n for n, q in h.named_parameters() if q.requires_grad and q.grad is None]
        assert not miss, miss
        assert all(torch.isfinite(q.grad).all() for q in h.parameters())
        if not bool(v.any()):
            assert float(L["loss_stop"]) == 0.0 and float(L["loss_acc"]) == 0.0 and float(L["loss_stop2s"]) == 0.0


def _straight_states(v: float = 5.0):
    """the executed LQR states of a straight draft at v m/s (every 2nd of the 41) and their times."""
    from navsim.agents.para_ssr.ck.v4 import lqr as LQR
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    traj = torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)[None]
    st = LQR.simulate(traj, torch.tensor([[v, 0.0]]), dtype=torch.float32)
    tt = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    return st[:, ::2], tt[::2]


def _car_code(n_q: int = 4) -> torch.Tensor:
    """det-frame code dims 0..7 of a car 8 m ahead, 1.5 m left of the ego axis (query 0); others far away."""
    code = torch.zeros(1, n_q, 8)
    code[0, :, 1] = 40.0
    code[0, 0] = torch.tensor([-1.5, 8.0, math.log(1.9), math.log(4.5), 0.0, math.log(1.6), 0.0, 1.0])
    return code


def _vehicle_cls(n_q: int = 4) -> torch.Tensor:
    cls = torch.full((1, n_q, 7), -6.0)
    cls[0, :, 0] = 4.0
    return cls


def test_v1c_relation_gradient_reaches_the_stop_branch():
    """design V2 + V1c: student rows built from the head's own final-layer deployed velocity carry the relation hazard's
    gradient into stop_branches[-1] and the velocity regression; a detached code (rel_det_grad off) carries none."""
    from navsim.agents.para_ssr.ck.online2 import _d_v5, student_rel_rows
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    rr = RelationReader(V5Cfg(relation="student", rel_ca=False))
    states, t = _straight_states()
    for grad in (True, False):
        h = ParaDetMotionHead(num_query=4, num_classes=7, embed_dims=16, bev_h=2, bev_w=4, code_size=10, fut_ts=2,
                              fut_mode=3, num_decoder_layers=2, num_heads=2, feedforward_channels=32, stop_branch=True)
        h.eval()
        o = h(_bev()[:1], return_hidden=True)
        code = torch.cat([_car_code(), o["all_bbox_preds"][:, :, :, 8:10][-1]], -1)      # dims 8, 9 keep the graph
        pred = {"all_bbox_preds": [code], "all_cls_scores": [_vehicle_cls()], "det_hidden": torch.zeros(1, 4, 256)}
        v5 = dict(_d_v5(), enabled=True, relation="student", rel_det_grad=grad)
        O = student_rel_rows(pred, v5)
        assert bool(O["valid"][0, 0]) and int(O["valid"].sum()) == 1
        out = rr(O, states, t, torch.zeros(1, dtype=torch.long))
        h.zero_grad()
        out["H"].sum().backward()
        w = h.stop_branches[-1].weight.grad
        if grad:
            assert w is not None and float(w.abs().sum()) > 0
            assert float(h.reg_branches[-1][-1].weight.grad[8:10].abs().sum()) > 0
            w0 = h.stop_branches[0].weight.grad                           # only the final layer feeds the rows
            assert w0 is None or float(w0.abs().sum()) == 0.0
        else:
            assert w is None or float(w.abs().sum()) == 0.0


def test_student_rows_read_the_gated_velocity():
    """a confidently stopped agent (stop logit 12): deployed velocity ~ 0, so v0 ~ 0 and the soft hold pi_s ~ sigmoid(2);
    the rows never carry an acceleration (critic mode 7 off: acc 0, has False)."""
    from navsim.agents.para_ssr.ck.v5.student_objects import student_object_rows
    raw = torch.tensor([[[-2.0, 0.5]] * 4])
    stop = torch.full((1, 4), 12.0)
    code = torch.cat([_car_code(), raw * (1 - torch.sigmoid(stop))[..., None]], -1)
    O = student_object_rows(code, _vehicle_cls(), torch.zeros(1, 4, 256), 96)
    assert float(O["sp0"][0, 0]) < 1e-4
    assert abs(float(O["pi_s"][0, 0]) - float(torch.sigmoid(torch.tensor(2.0)))) < 1e-3
    assert float(O["acc"].abs().sum()) == 0.0 and not bool(O["has"].any())
    O_open = student_object_rows(torch.cat([_car_code(), raw], -1), _vehicle_cls(), torch.zeros(1, 4, 256), 96)
    assert abs(float(O_open["sp0"][0, 0]) - math.hypot(2.0, 0.5)) < 1e-5 and float(O_open["pi_s"][0, 0]) < 1e-6


# ----------------------------------------------------------------------------------------------- ACC head: targets
def _scene(speeds, present, names, n_frames: int, cur: int = 3):
    """a SimpleNamespace scene: track j at (10 + 3 j, 1) m ahead in every frame where present[j][k] (k = f - cur; the
    history frames have every track), forward velocity speeds[j][k]."""
    frames = []
    for f in range(n_frames):
        k = f - cur
        rows, vel, nm, tok = [], [], [], []
        for j, (sp, pr, name) in enumerate(zip(speeds, present, names)):
            if k >= 0 and not pr[k]:
                continue
            rows.append([10.0 + 3 * j, 1.0, 0.5, 4.5, 1.9, 1.6, 0.0])
            vel.append([sp[max(k, 0)], 0.0, 0.0])
            nm.append(name)
            tok.append(f"trk{j}")
        ann = SimpleNamespace(boxes=np.asarray(rows, np.float32).reshape(-1, 7),
                              velocity_3d=np.asarray(vel, np.float32).reshape(-1, 3), names=nm, track_tokens=tok)
        frames.append(SimpleNamespace(annotations=ann, ego_status=SimpleNamespace(ego_pose=np.array([0.5 * k, 0, 0.0]))))
    return SimpleNamespace(frames=frames)


def _tcfg(**kw):
    return SimpleNamespace(max_agents=4, fut_ts=0, pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
                           det_fov_half_angle_deg=80.0, **kw)


def test_acc_targets_from_the_scene_keyframes():
    """keyframes k = 0..4 = frames cur..cur+4: a decelerating vehicle, a cone with an annotated 1 m/s (-> 0, static
    class), a pedestrian missing in frame cur+3 (absent there, later frames still read); a scene ending at cur+2 ->
    k >= 3 absent; off: no keys and the cache name of 1abf369; an unknown class is refused."""
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    speeds = [[4.0, 3.0, 1.0, 0.3, 0.0], [1.0] * 5, [1.2, 1.3, 1.1, 1.0, 0.9]]
    present = [[True] * 5, [True] * 5, [True, True, True, False, True]]
    names = ["vehicle", "traffic_cone", "pedestrian"]
    b = ParaSSRTargetBuilder(_tcfg(det_acc_head=True), trajectory_sampling=SimpleNamespace())
    tg = b._compute_agent_targets(_scene(speeds, present, names, 8), 3)
    slot = {int(round(float(tg["gt_boxes"][i, 1]))): i for i in range(3)}          # y_fwd 10 / 13 / 16 -> track
    sp, pr = tg["gt_kf_speed"].numpy(), tg["gt_kf_present"].numpy()
    assert sp.shape == pr.shape == (4, 5) and sp.dtype == np.float32 and pr.dtype == np.bool_
    np.testing.assert_allclose(sp[slot[10]], [4.0, 3.0, 1.0, 0.3, 0.0], atol=1e-6)
    assert pr[slot[10]].all() and pr[slot[13]].all() and (sp[slot[13]] == 0).all()
    assert pr[slot[16]].tolist() == [True, True, True, False, True]
    np.testing.assert_allclose(sp[slot[16]][[0, 1, 2, 4]], [1.2, 1.3, 1.1, 0.9], atol=1e-6)
    assert sp[slot[16], 3] == 0.0 and not pr[3].any() and not sp[3].any()               # slot 3: no GT
    short = b._compute_agent_targets(_scene(speeds, present, names, 6), 3)
    assert short["gt_kf_present"][:3, :3].all() and not short["gt_kf_present"][:, 3:].any()
    off = ParaSSRTargetBuilder(_tcfg(), trajectory_sampling=SimpleNamespace())._compute_agent_targets(
        _scene(speeds, present, names, 8), 3)
    assert set(off) == {"gt_boxes", "gt_labels", "gt_valid", "gt_fut_trajs", "gt_fut_masks"}
    for k in off:
        assert torch.equal(off[k], tg[k]), k
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    cfg = ParaSSRConfig()
    assert ParaSSRTargetBuilder(cfg, ts).get_unique_name() == HEAD_TARGET_NAME
    assert ParaSSRTargetBuilder(replace(cfg, det_acc_head=True), ts).get_unique_name() != HEAD_TARGET_NAME
    with pytest.raises(ValueError, match="unknown object class"):
        ParaSSRTargetBuilder._track_speeds(_scene([[1.0] * 5], [[True] * 5], ["unicorn"], 8), "trk0", 3, 5)


# ----------------------------------------------------------------------------------------------- ACC head: loss
def test_acc_loss_rules_and_values():
    """final-layer matches q i <-> GT i (q 4 unmatched).  GT 0: 4 -> 3 -> 1 -> 0.3 -> 0 (a = -2, stops); GT 1: k = 1
    missing (no a), never below 0.5 and not all present (stop2s unknown); GT 2: 2 -> 0.4 then gone (a = -3.2, stopped
    at k = 1: positive although the track ends); GT 3: 3 m/s throughout (a = 0, stop2s negative).  Huber (delta 1) x
    0.5 x 0.25 / 4 matches, focal x 0.2 / 4; the unmatched query and masked rows contribute nothing."""
    gt = torch.zeros(1, 4, 9)
    for i in range(4):
        gt[0, i] = torch.tensor([-6.0 + 4 * i, 6.0 + 5 * i, 0.0, 1.9, 4.5, 1.6, 0.0, 0.0, 2.0])
    lab = torch.zeros(1, 4, dtype=torch.long)
    val = torch.ones(1, 4, dtype=torch.bool)
    sp = torch.tensor([[[4.0, 3.0, 1.0, 0.3, 0.0], [5.0] * 5, [2.0, 0.4, 0.0, 0.0, 0.0], [3.0] * 5]])
    pr = torch.tensor([[[True] * 5, [True, False, True, True, True], [True, True, False, False, False], [True] * 5]])
    h = _small(5, acc_head=True)
    p = _synthetic_preds(h, gt, lab, seed=3)
    q, _ = hungarian_assign_det(p["all_cls_scores"][0, 0], p["all_bbox_preds"][0, 0], normalize_bbox(gt[0]), lab[0])
    assert q.tolist() == [0, 1, 2, 3]
    L = h.loss(p, gt, lab, val, gt_kf_speed=sp, gt_kf_present=pr)
    a = p["acc_preds"][0]
    d = a[[0, 2, 3]] - torch.tensor([-2.0, -3.2, 0.0])
    hub = torch.where(d.abs() < 1, 0.5 * d ** 2, d.abs() - 0.5)
    assert torch.allclose(L["loss_acc"], hub.sum() / 4 * 0.5 * 0.25, rtol=1e-6, atol=0)
    s = p["stop2s_logits"][0, [0, 2, 3]]
    ref = _focal(s, torch.tensor([True, True, False])).sum() / 4 * 0.2
    assert torch.allclose(L["loss_stop2s"], ref, rtol=1e-6, atol=0)
    p2 = dict(p, acc_preds=p["acc_preds"].clone(), stop2s_logits=p["stop2s_logits"].clone())
    p2["acc_preds"][0, [1, 4]] += 7.0
    p2["stop2s_logits"][0, [1, 4]] -= 9.0
    L2 = h.loss(p2, gt, lab, val, gt_kf_speed=sp, gt_kf_present=pr)
    assert float(L2["loss_acc"]) == float(L["loss_acc"]) and float(L2["loss_stop2s"]) == float(L["loss_stop2s"])
    # the other det terms are those of the plain head
    plain = _small(5).loss({k: p[k] for k in ("all_cls_scores", "all_bbox_preds", "traj_preds", "traj_cls_preds")},
                           gt, lab, val)
    assert all(float(plain[k]) == float(L[k]) for k in plain)
    with pytest.raises(KeyError, match="gt_kf_speed"):
        h.loss(p, gt, lab, val)
    with pytest.raises(ValueError, match="must be"):
        h.loss(p, gt, lab, val, gt_kf_speed=sp[..., :4], gt_kf_present=pr[..., :4])


# ----------------------------------------------------------------------------------------------- model level
def _tiny_cfg(**kw):
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    return ParaSSRConfig(use_lidar=False, image_architecture="resnet18", backbone_pretrained=False, image_scale=1.0,
                         crop_top=0, bev_h=8, bev_w=8, embed_dims=16, num_heads=2, ffn_channels=32,
                         encoder_num_layers=1, encoder_num_points_in_pillar=2, encoder_num_points_sca=2,
                         encoder_attn_dropout=0.0, encoder_ffn_dropout=0.0, latent_num_layers=1, num_scenes=2,
                         num_query=4, max_agents=4, det_num_decoder_layers=1, map_num_vec=3, map_num_pts_per_vec=4,
                         map_num_decoder_layers=1, map_max_vec=3, map_num_orders=2, use_grid_mask=False,
                         plan_anchor=False, **kw)


def test_model_and_loss_with_the_det_options():
    """ParaSSRModel + ParaSSRLoss with V1b / V1c / ACC on: every existing parameter of the whole model equals the
    off model's (seeded), an off checkpoint does not load strictly (the new keys are missing), the new det loss keys
    reach the det task, every det-head parameter gets a gradient, missing ACC targets are refused, and eval-mode
    predictions with expose_det_outputs carry the stop logits while the aux decoder reads the gated velocity."""
    from navsim.agents.para_ssr.para_ssr_features import ParaSSRFeatureBuilder
    from navsim.agents.para_ssr.para_ssr_loss import ParaSSRLoss
    from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel
    from navsim.evaluate.aux_metrics import _stable_flat_topk, decode_detection_predictions
    sys.path.insert(0, str(REPO / "tests"))
    from test_para_ssr_front_cameras import _front_input
    opts = dict(det_vel_moving_only=True, det_stop_branch=True, det_acc_head=True, expose_det_outputs=True)
    torch.manual_seed(9)
    m_off = ParaSSRModel(_tiny_cfg())
    torch.manual_seed(9)
    cfg = _tiny_cfg(**opts)
    m_on = ParaSSRModel(cfg).train()
    so, sn = m_off.state_dict(), m_on.state_dict()
    assert all(torch.equal(so[k], sn[k]) for k in so)
    new = set(sn) - set(so)
    assert new and all(k.startswith(("det_motion_head.stop_branches.", "det_motion_head.acc_head.")) for k in new)
    with pytest.raises(RuntimeError, match="Missing key"):
        m_on.load_state_dict(so, strict=True)
    feats = {n: t.unsqueeze(0) for n, t in ParaSSRFeatureBuilder(cfg).compute_features(_front_input()).items()}
    pr = m_on(feats)
    gt, lab, val, _, _ = _gt()
    sp, pk = _kf()
    tg = {"trajectory_offsets": torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(3)),
          "trajectory_mask": torch.ones(1, 8), "command": torch.tensor([[1.0, 0, 0, 0]]),
          "gt_boxes": gt[:1, :4], "gt_labels": lab[:1, :4], "gt_valid": val[:1, :4],
          "gt_fut_trajs": torch.zeros(1, 4, 8, 2), "gt_fut_masks": torch.ones(1, 4, 8),
          "gt_map_pts": torch.rand(1, 3, 2, 4, 2, generator=torch.Generator().manual_seed(6)) * 20,
          "gt_map_labels": torch.tensor([[0, 2, 0]]), "gt_map_valid": torch.tensor([[True, True, False]])}
    loss_fn = ParaSSRLoss(cfg)
    with pytest.raises(KeyError, match="gt_kf_speed"):
        loss_fn(m_on, feats, tg, pr)
    tg.update(gt_kf_speed=sp[:1, :4], gt_kf_present=pk[:1, :4])
    total, logs = loss_fn(m_on, feats, tg, pr)
    assert {"loss_stop", "loss_acc", "loss_stop2s"} <= set(logs)
    assert all(math.isfinite(float(logs[k])) for k in ("loss_stop", "loss_acc", "loss_stop2s"))
    total.backward()
    miss = [n for n, q in m_on.det_motion_head.named_parameters() if q.requires_grad and q.grad is None]
    assert not miss, miss
    m_on.eval()
    with torch.no_grad():
        pe = m_on(feats)
    assert {"all_stop_logits", "all_vel_raw", "acc_preds", "stop2s_logits", "all_bbox_preds"} <= set(pe)
    dec = decode_detection_predictions(pe, max_predictions=5)[0]
    sel = _stable_flat_topk(pe["all_cls_scores"][-1][0].sigmoid().numpy().reshape(-1), 5) // 7
    gated = pe["all_bbox_preds"][-1][0, sel, 8:10].numpy()
    np.testing.assert_array_equal(dec["boxes"][:, 7:9], gated)
    raw = pe["all_vel_raw"][-1][0, sel].numpy()
    assert not np.array_equal(raw, gated)


# ----------------------------------------------------------------------------------------------- p_stop_i (rel_pstop)
def test_rel_pstop_feature_and_refusals():
    """V5Cfg.rel_pstop: off = the reliability MLP of N_FEAT_S inputs (state dict unchanged); on = one more input whose
    column is sigmoid(stop logit) of the selected queries (0 on padding, detached); teacher relation refuses it;
    student_rel_rows / the e2e config refuse it without the det head's stop logits."""
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, _d_v5, student_rel_rows
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    from navsim.agents.para_ssr.ck.v5.student_objects import N_FEAT_S, student_object_rows
    assert V5Cfg().rel_pstop is False and V5Cfg.from_dict({"relation": "student"}).rel_pstop is False
    torch.manual_seed(4)
    a = RelationReader(V5Cfg(relation="student", rel_ca=False))
    torch.manual_seed(4)
    b = RelationReader(V5Cfg(relation="student", rel_ca=False, rel_pstop=True))
    assert a.rel_mlp[0].in_features == N_FEAT_S and b.rel_mlp[0].in_features == N_FEAT_S + 1
    assert set(a.state_dict()) == set(b.state_dict())
    with pytest.raises(ValueError, match="rel_pstop"):
        V5Cfg(relation="teacher", rel_pstop=True).validate()
    code = torch.cat([_car_code(), torch.tensor([[[-2.0, 0.0]] * 4])], -1)
    code[0, 1, :8] = code[0, 0, :8]
    code[0, 1, 0] = 2.5                                              # a second in-window car
    stop = torch.tensor([[1.5, -0.5, 3.0, 0.0]], requires_grad=True)
    O = student_object_rows(code, _vehicle_cls(), torch.zeros(1, 4, 256), 4, stop_logit=stop)
    O0 = student_object_rows(code, _vehicle_cls(), torch.zeros(1, 4, 256), 4)
    assert O["feats"].shape[-1] == N_FEAT_S + 1 and O0["feats"].shape[-1] == N_FEAT_S
    assert torch.equal(O["feats"][..., :-1], O0["feats"]) and not O["feats"].requires_grad
    qi = O["q_idx"][0]
    want = torch.where(qi >= 0, torch.sigmoid(stop.detach()[0, qi.clamp(min=0)]), torch.zeros(()))
    assert torch.equal(O["feats"][0, :, -1], want) and int((qi >= 0).sum()) == 2 and O["feats"].shape[1] == 2
    assert b.reliability(O).shape == (1, 2)
    v5 = dict(_d_v5(), enabled=True, relation="student", rel_pstop=True)
    pred = {"all_bbox_preds": [code], "all_cls_scores": [_vehicle_cls()], "det_hidden": torch.zeros(1, 4, 256)}
    with pytest.raises(KeyError, match="det_stop_branch"):
        student_rel_rows(pred, v5)
    O2 = student_rel_rows(dict(pred, all_stop_logits=[stop]), v5)
    assert torch.equal(O2["feats"], O["feats"])
    with pytest.raises(ValueError, match="rel_pstop"):
        CKE2E2Config.from_any({"v4": {"v5": {"enabled": True, "relation": "off", "rel_pstop": True}}})
    c = CKE2E2Config.from_any({"enabled": True, "v4": {"v5": {"enabled": True, "relation": "student",
                                                              "rel_pstop": True}}})
    assert c.v5_net().rel_pstop is True and c.v5_net().rel_ca is False
    parent = SimpleNamespace(refiner_mode="off", ck_e2e=None, plan_anchor=True, bev_h=50, bev_w=100,
                             pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0), embed_dims=256, det_stop_branch=False)
    with pytest.raises(ValueError, match="det_stop_branch"):
        c.validate(parent)
    with pytest.raises(ValueError) as e:                             # passes this check, fails on a later one
        c.validate(SimpleNamespace(**dict(vars(parent), det_stop_branch=True)))
    assert "det_stop_branch" not in str(e.value)
    assert CKE2E2Config.from_any({}).v5["rel_pstop"] is False


def test_mode7_stays_off_with_the_acc_head():
    """the ACC head's outputs feed nothing in the critic: the student rows are the same with and without acc_preds /
    stop2s_logits in the predictions (acc 0, has False), the student V5Cfg keeps rel_ca off."""
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, _d_v5, student_rel_rows
    code = torch.cat([_car_code(), torch.tensor([[[-2.0, 1.0]] * 4])], -1)
    pred = {"all_bbox_preds": [code], "all_cls_scores": [_vehicle_cls()], "det_hidden": torch.zeros(1, 4, 256)}
    v5 = dict(_d_v5(), enabled=True, relation="student")
    A = student_rel_rows(pred, v5)
    B = student_rel_rows(dict(pred, acc_preds=torch.full((1, 4), -3.0), stop2s_logits=torch.full((1, 4), 5.0)), v5)
    assert set(A) == set(B) and all(torch.equal(A[k], B[k]) for k in A)
    assert float(A["acc"].abs().sum()) == 0 and not bool(A["has"].any())
    c = CKE2E2Config.from_any({"v4": {"v5": {"enabled": True, "relation": "student"}}})
    assert c.v5_net().rel_ca is False


# ----------------------------------------------------------------------------------------------- eval / arm files
def test_infer_check_v5_student_with_pstop(tmp_path):
    """eval_e2e2: a rel_pstop student's infer_check rebuilds the rows from check_bev.npz's det inputs incl. det_stop and
    replays v5_traj exactly; a dump without det_stop is refused."""
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import (CKE2E2Config, build_student_ck2, infer_pool, student_rel_rows)
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                                     "n_obj": 8, "rel_pstop": True}}
    cfg = CKE2E2Config.from_any({"v4": v4})
    torch.manual_seed(0)
    net = build_student_ck2(cfg, apply_prior=False)
    assert net.relation.rel_mlp[0].in_features == 271
    R = tmp_path / "e2e_v5"
    ckpt = R / "train" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {E1.STUDENT_PREFIX + k: v for k, v in net.state_dict().items()}}, ckpt)
    P = E1.Paths(R, "navtrain_val")
    E2.extract_ck2(ckpt, P.ck_run, "e2e_v5", {"enabled": True, "seed": 0, "v4": v4})
    net2, _ = load_ck(P.ck_run, "last", "cpu")
    assert net2.v5.rel_pstop and net2.relation.rel_mlp[0].in_features == 271
    n, Q = 2, 12
    anc = torch.from_numpy(np.asarray(AS.load_anchors(), np.float32))
    top = torch.stack([anc[i * 16:(i + 1) * 16] for i in range(n)])
    scale = torch.tensor([1.0, 0.9, 0.95, 1.05, 1.0, 1.0]).reshape(1, 1, 6, 1, 1)
    raw = (top[:, :, None] * scale).reshape(n, 96, 8, 3).contiguous()
    status = torch.zeros(n, 8)
    status[:, 1], status[:, 4] = 1.0, 6.0
    P.packed.mkdir(parents=True, exist_ok=True)
    for k, v in (("cand96", raw), ("status", status), ("valid96", torch.ones(n, 96, dtype=torch.bool)),
                 ("v2_final", torch.linspace(1, 0, 16).expand(n, 16)), ("v2_im", torch.full((n, 16), 1 / 16))):
        np.save(P.packed / f"{k}.npy", v.numpy())
    g = torch.Generator().manual_seed(1)
    bev = torch.randn(n, 256, 50, 100, generator=g)
    code = torch.zeros(n, Q, 10)
    code[..., 0] = torch.rand(n, Q, generator=g) * 8 - 4
    code[..., 1] = torch.rand(n, Q, generator=g) * 25 + 3
    code[..., 2:4] = 0.5
    code[..., 7] = 1.0
    cls = torch.randn(n, Q, 7, generator=g) + 1.0
    hid = torch.randn(n, Q, 256, generator=g)
    stop = torch.randn(n, Q, generator=g) * 3
    rows = np.arange(n)
    P.infer.mkdir(parents=True, exist_ok=True)
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy(), det_code=code.numpy(), det_cls=cls.numpy(),
             det_hidden=hid.numpy(), det_stop=stop.numpy())
    cand, fin, par = infer_pool(raw, raw[:, ::6])
    ego = ego_inputs(status)
    with torch.no_grad():
        feat, mem = net2.trunk.scene(bev, n)
        pref = net2.scene_pref(mem, ego)
        rel = student_rel_rows({"all_bbox_preds": [code], "all_cls_scores": [cls], "det_hidden": hid,
                                "all_stop_logits": [stop]}, cfg.v5)
        o = net2.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=rel)
        src = torch.tensor([3, 50])
        rf = net2.refine_drafts(o, cand.float(), src[:, None], ego)
        from navsim.agents.para_ssr.ck.online2 import lateral_decode
        lat = lateral_decode(cand, o["w_lat"], ego[0], 0.0)
    for k, v in (("score_logit", o["score_logit"]), ("w_lat", o["w_lat"]), ("e_lat", lat["e_lat"][..., 2:]),
                 ("lat_traj", lat["traj"]), ("v5_traj", rf["tau1"]), ("v5_mu", rf["mu"][:, 0])):
        np.save(P.infer / f"{k}.npy", v.float().numpy())
    np.save(P.infer / "v5_src.npy", src.numpy().astype(np.int16))
    a = SimpleNamespace(gpus="")
    res = E2.stage_infer_check(P, a)
    assert res["pass"] and res["max_abs_err"]["v5_traj"] < 1e-5
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy(), det_code=code.numpy(), det_cls=cls.numpy(),
             det_hidden=hid.numpy())
    with pytest.raises(SystemExit, match="det_stop"):
        E2.stage_infer_check(P, a)


def test_arm_files_switch_the_det_fixes_on():
    """fusion_v13_v5.extra (arms B and D) turns V1b / V1c / the ACC head on; v5_arm.set turns rel_pstop on, and the
    launcher's set application accepts it; nothing turns the critic's mode 7 (rel_ca) or V3 / V4 on."""
    from tools.ck.e2e2 import launch_util2 as LU2
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    ex = (REPO / "tools/ck/e2e2/fusion_v13_v5.extra").read_text().splitlines()
    for k in ("det_vel_moving_only", "det_stop_branch", "det_acc_head"):
        assert f"++agent.config.{k}=true" in ex, k
    assert "++agent.config.det_vel_code_weights=[1.05,0.41]" in ex
    sets = [(REPO / f"tools/ck/e2e2/{f}").read_text().splitlines() for f in ("v5_arm.set", "v5kd_arm.set")]
    assert "v4.v5.rel_pstop=true" in sets[0]
    lines = [x for s in sets for x in s]
    assert not any(x.split("=")[0].strip().endswith("rel_ca") for x in lines if not x.startswith("#"))
    # the 'rel' group reads the det head's futures (rel_modes cv+fut); mode 7 (cv+fut+ca) stays off
    assert not any(x.split("=")[0].strip().endswith("rel_modes") and "ca" in x.split("=", 1)[1] for x in lines
                   if not x.startswith("#"))
    for arm in (sets[0], sets[0] + sets[1]):
        ck = LU2.apply_sets2(LU2.load_ck2_yaml(), arm)
        v5 = CKE2E2Config.from_any(ck).v5
        assert v5["rel_pstop"] is True and v5["relation"] == "student" and v5["rel_det_grad"] is True
        assert CKE2E2Config.from_any(ck).v5_net().rel_ca is False
