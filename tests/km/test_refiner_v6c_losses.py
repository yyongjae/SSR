"""refiner v6c student-side losses, labels, warm-up draws and logs after report 129 (package P3; CPU).

  * F10 (A19) comfort_hinge = hinges on the official ego_is_comfortable features: 0 <=> exact_comfort == 1
  * F11 (A20) front hinges shift by the state's own speed (as the official TTC), not the lagged displacement speed
  * F12 (N-stride2-odd-valid) stride2's valid_bin keeps an odd last observed step's positive in the BCE
  * F13 (A39) the continuous L_E clips only the target (an over-shooting output gets a gradient back)
  * F15 (A40 / A73) warm-up draws: seed iteration * world + rank, subset and mid anchors drawn per sample
  * F16 (A72) the warm-up returns the mixture loss without the on-policy rows' official / label / teacher work
  * F17 (A64, A66, A69, A70) lon_noop, Eskill / Eprior, over-labelling, step-exact *_sum keys, KDcov every call
  * F18 (A51) the eval pick needs plan_select_index
The agent tests use the 2-token fixture of test_refiner_v6c_agent.py and are skipped when its inputs are absent.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.modules import exact_verdict as EV
from navsim.agents.para_ssr.modules.torch_tracker import _track, track
from navsim.agents.para_ssr.refiner_v6c import hook as H
from navsim.agents.para_ssr.refiner_v6c import labels as L
from navsim.agents.para_ssr.refiner_v6c import losses as LS
from navsim.agents.para_ssr.refiner_v6c import mixture as MX
from navsim.agents.para_ssr.refiner_v6c import nets as N

sys.path.insert(0, os.path.dirname(__file__))
from test_refiner_v6c_agent import FIX, _fresh, build_agent, have_inputs, make_fixture  # noqa: E402

ANCHORS = "resources/km_corridor/trajectory_anchors_256.npy"


def _anchors(dtype=torch.float64):
    return torch.tensor(np.load(ANCHORS), dtype=dtype)


# ------------------------------------------------------------------------------------------------- F10 comfort
def _anchor_grid():
    """256 raw anchors x 12 inits (vx0 1 / 4 / 8 / 12 m/s, ax0 -1 / 0 / +1 m/s^2), tracked (float64)"""
    anc = _anchors()
    ini = torch.tensor([[v, 0, a, 0, 0, 0, 0] for v in (1.0, 4.0, 8.0, 12.0) for a in (-1.0, 0.0, 1.0)],
                       dtype=torch.float64)
    poses = anc[None].expand(len(ini), -1, -1, -1).reshape(-1, 8, 3)
    init = ini[:, None].expand(-1, anc.shape[0], -1).reshape(-1, 7)
    return _track(poses, init, full=True)                                                    # [3072, 41, 7]


def test_comfort_hinge_zero_iff_officially_comfortable():
    S = _anchor_grid()
    ok = EV.exact_comfort(S)
    assert 0.2 < float(ok.mean()) < 0.9                                                       # both classes present
    h0 = LS.comfort_hinge(S, yaw_mask_speed=0.0)                                              # the official rule
    assert torch.equal(h0 == 0, ok == 1)
    h = LS.comfort_hinge(S)                                                                   # low-speed yaw mask
    assert bool((ok[h > 0] == 0).all()) and bool((h <= h0).all())                            # the mask only drops terms
    assert torch.equal(LS.comfort_hinge(S.float(), 0.0) == 0, EV.exact_comfort(S.float()) == 1)


def test_comfort_hinge_gradient_finite_at_zero_acceleration():
    """decode_v6 controls -> LQR with grad -> comfort hinge: finite gradient, also through the states with
    ax = ay = 0 (a stopped draft; hypot's gradient would be NaN there) and non-zero where the hinge is active"""
    anc = _anchors(torch.float32)
    tau = torch.cat([anc[[5, 100, 200]], torch.zeros(1, 8, 3)], 0)
    v0 = torch.tensor([8.0, 4.0, 12.0, 0.0])
    init = torch.zeros(4, 7)
    init[:, 0] = v0
    ctrl = torch.zeros(4, 12)
    ctrl[:3, :6], ctrl[:3, 6:] = -1.5, 1.0
    ctrl.requires_grad_()
    S = track(N.decode_v6(tau, ctrl, v0, mode="B")["traj"], init, full=True)
    assert int(((S[3, 1:, 5] == 0) & (S[3, 1:, 6] == 0)).sum()) == 40 and S.requires_grad
    h = LS.comfort_hinge(S, 0.0)
    assert float(h[:3].detach().max()) > 0
    g, = torch.autograd.grad(h.sum(), ctrl)
    assert torch.isfinite(g).all() and float(g[h > 0].abs().sum()) > 0


# ------------------------------------------------------------------------------------------------- F11 own speed
def _brake_and_accel():
    tt = torch.arange(1, 9, dtype=torch.float64) * 0.5
    z = torch.zeros_like(tt)
    acc = torch.stack([tt ** 2, z, z], -1)                                                     # 2 m/s^2 from rest
    brk = torch.stack([10 * tt - tt ** 2, z, z], -1)                                           # 10 m/s, -2 m/s^2
    init = torch.tensor([[0.0, 0, 1.0, 0, 0, 0, 0], [10.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    return track(torch.stack([acc, brk]), init, full=True).float()


def test_front_points_use_the_states_own_speed():
    S = _brake_and_accel()
    _, _, _, spd = LS._front_points(S)
    own = torch.hypot(S[:, 1:, 3], S[:, 1:, 4])
    assert torch.equal(spd, own)
    x0 = torch.cat([torch.zeros(2, 1), S[:, 1:, 0]], -1)
    lag = x0.diff(dim=-1) / 0.1                                                         # the old displacement speed
    assert float((own - lag).abs().max()) > 0.1
    # from v0 = 0 (ax0 > 0) state 1 already moves: moving follows its own speed, the lagged one is still 0
    assert float(S[0, 1, 3]) >= 5e-3 and bool(spd[0, 0] >= 5e-3) and float(lag[0, 0]) == 0.0


def test_ttc_hinge_fires_at_the_own_speed_shift_only():
    """constructed: positions held at x = 0 (lagged speed 0) while the state's own speed is 10 m/s -> the TTC shift
    (3 m at 0.3 s) reaches an object at FRONT + 1..2 m that the unshifted footprint never touches"""
    from navsim.agents.para_ssr.modules import corridor_head as CH
    S = torch.zeros(1, 41, 7)
    S[:, :, 3] = 10.0
    occ = torch.zeros(1, 49, 200, 200)
    xs = (torch.arange(200) + 0.5) * 0.32
    occ[:, :, (xs >= CH.FRONT + 1.0) & (xs <= CH.FRONT + 2.0), 90:110] = 1.0
    h = LS.front_hinge(S, occ, "prob", 0.3)
    assert float(h.min()) > 0                                                                  # every step: TTC part
    S_lag = S.clone()
    S_lag[:, :, 3] = 0.0                                                                       # own = lagged = 0
    assert float(LS.front_hinge(S_lag, occ, "prob", 0.3).abs().sum()) == 0.0
    S_far = S.clone()
    S_far[:, :, 3] = 1.0                                                                       # 0.3..0.9 m: short
    assert float(LS.front_hinge(S_far, occ, "prob", 0.3).abs().sum()) == 0.0


# ------------------------------------------------------------------------------------------------- F12 valid_bin
def _lab41(nobs, pos_step=None, r=1):
    steps = torch.arange(41)
    z = torch.zeros(1, r, 41)
    lab = {"ovl": z.clone(), "ttc": z.clone(), "offroad": z.clone(), "gap": torch.full_like(z, 20.0),
           "vapp": z.clone(), "pos": torch.zeros(1, r, 41, 2), "obj": z.clone(),
           "valid": (steps <= nobs).float().expand(1, r, 41).clone(), "valid_off": torch.ones(1, r, 41)}
    lab["pos"][..., 0] = 20.0
    if pos_step is not None:
        lab["ovl"][..., pos_step] = 1.0
    return lab


def _old_risk_loss(E, lab, kd=None, lam=None, pw=4.0):
    """the HEAD (2e674b4) risk_loss: 'valid' for ovl / ttc, cont_norm (clip) on both sides of the continuous terms"""
    v = lab["valid"]
    nv = v.sum().clamp(min=1.0)
    loss = E.new_zeros(())
    for nm, ch in LS.LOGIT_LABELS.items():
        vm = lab["valid_off"] if nm == "offroad" else v
        loss = loss + (LS._bce_pw(E[..., ch], lab[nm], pw) * vm).sum() / vm.sum().clamp(min=1.0)
    y = torch.stack([lab["gap"], lab["vapp"], lab["pos"][..., 0], lab["pos"][..., 1]], -1)
    lc = F.smooth_l1_loss(N.cont_norm(E[..., 2:6]), N.cont_norm(y), beta=0.5, reduction="none")
    for j in range(4):
        loss = loss + (lc[..., j] * v).sum() / nv
    if kd is not None:
        p, g = kd["p"], kd["g"]
        for ch, nm in enumerate(N.E_NAMES):
            w, gc = float(lam.get(nm, 0.0)), g[..., ch]
            if w <= 0 or not bool(gc.any()):
                continue
            ng = gc.sum().clamp(min=1.0)
            if ch in (0, 1, 6):
                l = F.binary_cross_entropy_with_logits(E[..., ch], p[..., ch].clamp(0, 1), reduction="none")
                l = (l * gc).sum() / ng
            else:
                l = (F.smooth_l1_loss(N.cont_norm(E[..., 2:6])[..., ch - 2], N.cont_norm(p[..., 2:6])[..., ch - 2],
                                      beta=0.5, reduction="none") * gc).sum() / ng
            loss = loss + w * l
    return loss


def test_stride2_valid_bin_keeps_an_odd_last_observed_positive():
    """report 127 §3.3: nobs 7 with an overlap at step 7 -> slot 4 = (0.6, 0.8] carries it with valid_bin 1, while
    'valid' (step 8) is 0: the BCE now weighs it, the continuous channels at slot 4 stay masked"""
    s = L.stride2(_lab41(7, pos_step=7))
    assert float(s["ovl"][0, 0, 4]) == 1.0 and float(s["valid_bin"][0, 0, 4]) == 1.0
    assert float(s["valid"][0, 0, 4]) == 0.0 and float(s["valid_bin"][0, 0, 5]) == 0.0
    E = torch.zeros(1, 1, 21, 12, requires_grad=True)
    l, _ = LS.risk_loss(E, s)
    g, = torch.autograd.grad(l, E)
    assert float(g[0, 0, 4, 0]) != 0.0                                                         # ovl BCE weight > 0
    assert float(g[0, 0, 4, 2:6].abs().sum()) == 0.0 and float(g[0, 0, 3, 2:6].abs().sum()) > 0
    s_old = {k: v for k, v in s.items() if k != "valid_bin"}                                   # the old masking
    g_old, = torch.autograd.grad(LS.risk_loss(E, s_old)[0], E)
    assert float(g_old[0, 0, 4, 0]) == 0.0


def test_risk_loss_bitwise_old_for_an_even_horizon():
    torch.manual_seed(0)
    lab = L.stride2(_lab41(8, pos_step=8, r=3))
    assert torch.equal(lab["valid_bin"], lab["valid"])
    lab["gap"] = torch.rand(1, 3, 21) * 30
    E = (torch.randn(1, 3, 21, 12) * 2).requires_grad_()                                       # |o| <= 4 on 2:6
    kd = {"p": torch.cat([torch.rand(1, 3, 21, 2), torch.randn(1, 3, 21, 4) * 10, torch.rand(1, 3, 21, 1)], -1),
          "g": (torch.rand(1, 3, 21, 7) > 0.3).float()}
    lam = {nm: 0.5 for nm in N.E_NAMES}
    for k, l_ in ((None, None), (kd, lam)):
        ln, _ = LS.risk_loss(E, lab, k, l_)
        lo = _old_risk_loss(E, lab, k, l_)
        gn, = torch.autograd.grad(ln, E)
        go, = torch.autograd.grad(lo, E)
        assert torch.equal(ln, lo) and torch.equal(gn, go)
    assert "valid_bin" not in L.stride2({"ovl": torch.zeros(1, 1, 41)})                        # no 'valid': unchanged


# ------------------------------------------------------------------------------------------------- F13 no output clip
def _cont_case(o, gap=20.0, px=20.0):
    n = len(o)
    E = torch.zeros(1, 1, n, 12)
    E[..., 2] = torch.tensor(o) * N.CONT_NORM[0]
    E[..., 4] = torch.tensor(o) * N.CONT_NORM[2]
    lab = {"ovl": torch.zeros(1, 1, n), "ttc": torch.zeros(1, 1, n), "offroad": torch.zeros(1, 1, n),
           "gap": torch.full((1, 1, n), gap), "vapp": torch.zeros(1, 1, n),
           "pos": torch.stack([torch.full((1, 1, n), px), torch.zeros(1, 1, n)], -1), "obj": torch.ones(1, 1, n),
           "valid": torch.ones(1, 1, n), "valid_off": torch.ones(1, 1, n)}
    return E.requires_grad_(), lab


def test_continuous_loss_clips_only_the_target():
    E, lab = _cont_case([1.0, 4.5, 6.0], gap=20.0)
    g, = torch.autograd.grad(LS.risk_loss(E, lab)[0], E)
    assert bool((g[0, 0, :, 2] != 0).all())                                                    # all three pulled
    assert float(g[0, 0, 0, 2]) < 0 and float(g[0, 0, 1, 2]) > 0 and float(g[0, 0, 2, 2]) > 0
    go, = torch.autograd.grad(_old_risk_loss(E, lab), E)
    assert float(go[0, 0, 0, 2]) != 0 and float(go[0, 0, 1:, 2].abs().sum()) == 0.0              # old: clamp killed it
    E2, lab2 = _cont_case([4.5], px=64.0)                                                       # target clipped to 4
    g2, = torch.autograd.grad(LS.risk_loss(E2, lab2)[0], E2)
    assert float(g2[0, 0, 0, 4]) > 0                                                           # descent -> toward 4
    E3, lab3 = _cont_case([-3.9, -1.0, 0.0, 2.5, 4.0], gap=15.0, px=-30.0)                     # |o| <= 4: unchanged
    ln, lo = LS.risk_loss(E3, lab3)[0], _old_risk_loss(E3, lab3)
    assert torch.equal(ln, lo)
    assert torch.equal(torch.autograd.grad(ln, E3)[0], torch.autograd.grad(lo, E3)[0])


# ------------------------------------------------------------------------------------------------- F17 risk logs
def test_eskill_and_kdcov_logs():
    torch.manual_seed(0)
    lab = L.stride2(_lab41(40, r=50))
    lab["ovl"] = (torch.rand(1, 50, 21) < 0.1).float()
    lab["ttc"] = lab["ovl"].clone()
    lab["offroad"] = lab["ovl"].clone()
    pr = float((lab["ovl"] * lab["valid_bin"]).sum() / lab["valid_bin"].sum())
    q = 4 * pr / (4 * pr + 1 - pr)
    E = torch.zeros(1, 50, 21, 12)
    E[..., [0, 1, 6]] = float(np.log(q / (1 - q)))                                              # best constant logit
    _, lg = LS.risk_loss(E, lab)
    assert abs(float(lg["v6c/Eskill_ovl"])) < 1e-4 and float(lg["v6c/Eprior_ovl"]) > 0
    E[..., [0, 1, 6]] = (lab["ovl"][..., None] * 2 - 1) * 12.0                                 # near-perfect
    _, lg = LS.risk_loss(E, lab)
    assert float(lg["v6c/Eskill_ttc"]) > 0.99 and float(lg["v6c/Eskill_offroad"]) > 0.99
    # KDcov on every call, also when the gate is empty or the weight 0; KD_* only for live channels
    kd = {"p": torch.rand(1, 50, 21, 7), "g": torch.zeros(1, 50, 21, 7)}
    kd["g"][..., 2] = 1.0
    lam = {nm: 0.5 for nm in N.E_NAMES}
    lam["vapp"] = 0.0
    _, lg = LS.risk_loss(E, lab, kd, lam)
    assert all(f"v6c/KDcov_{nm}" in lg for nm in N.E_NAMES)
    assert float(lg["v6c/KDcov_gap"]) == 1.0 and float(lg["v6c/KDcov_ovl"]) == 0.0
    assert set(k for k in lg if k.startswith("v6c/KD_")) == {"v6c/KD_gap"}


def test_cond_and_overlabel_logs():
    logs = {}
    x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    H._cond(logs, "k", x, torch.tensor([True, False, True, False]))
    assert float(logs["k"]) == 3.0 and float(logs["k_sum"]) / 0.5 == 3.0
    logs = {}
    H._cond(logs, "k", x, torch.zeros(4, dtype=torch.bool))
    assert set(logs) == {"k_sum"} and float(logs["k_sum"]) == 0.0
    # rows [B=2, R=2]: (0, 0) TTC step positive but official TTC passes -> over-labelled; (0, 1) positive and failed;
    # (1, 0) no positive; (1, 1) positive but no table (NaN)
    ttc = torch.zeros(2, 2, 21)
    ttc[0, 0, 5] = ttc[0, 1, 9] = ttc[1, 1, 3] = 1.0
    lab = {"ttc": ttc, "ovl": torch.zeros(2, 2, 21)}
    ex = {"ttc1": torch.tensor([[1.0, 0.0], [1.0, float("nan")]]), "nc1": torch.ones(2, 2)}
    logs = {}
    H.overlabel_logs(logs, lab, ex)
    assert float(logs["v6c/ttc_overlabel"]) == 0.5 and float(logs["v6c/ttc_steppos"]) == 0.5
    assert abs(float(logs["v6c/ttc_overlabel_sum"]) / float(logs["v6c/ttc_steppos"]) - 0.5) < 1e-7
    assert "v6c/ovl_overlabel" not in logs and float(logs["v6c/ovl_overlabel_sum"]) == 0.0
    assert float(logs["v6c/ovl_steppos"]) == 0.0


def test_lon_noop_flag():
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    moving = torch.stack([10 * tt, 0.02 * tt ** 2, 0.004 * tt], -1)[None]
    stopped = torch.zeros(1, 8, 3)                                                             # stopped from pose 1

    def noop(tau, v0, z, mode="A"):
        ctrl = torch.zeros(1, 12)
        ctrl[:, :6] = z
        return float(H.lon_noop(N.decode_v6(tau, ctrl, torch.tensor([v0]), mode=mode)))
    assert noop(moving, 10.0, 0.0) == 1.0 and noop(moving, 10.0, 0.0, "B") == 1.0              # ctrl = 0
    assert noop(stopped, 0.0, -1.0) == 1.0                                                     # decel, already stopped
    assert noop(moving, 10.0, -1.0) == 0.0                                                     # a live decel
    assert noop(moving, 10.0, 1.0) == 1.0 and noop(moving, 10.0, 1.0, "B") == 0.0             # mode-A clamp / B live


# ------------------------------------------------------------------------------------------------- F15 warm-up draws
def _stub(b=4, it=7, same_hum=True):
    torch.manual_seed(1)
    anc = _anchors(torch.float32)
    hum = anc[[30]].repeat(b, 1, 1) if same_hum else anc[torch.arange(b) * 9]
    return ({"trajectory_anchors": anc, "_v6c_iteration": it},
            {"ex2_hum": hum + 0.05, "ex2_hvalid": torch.ones(b)})


def _old_path_mixture(anchors, hum, hvalid, gen, n_anchor=32, n_rule=32, n_gt=16):
    """HEAD (2e674b4) path_mixture: one batch-shared mid draw"""
    b, k = hum.shape[0], anchors.shape[0]
    order = torch.cdist(hum.reshape(b, -1), anchors.reshape(k, -1).to(hum)).argsort(-1)
    near = order[:, :8]
    mid = order[:, 8 + torch.randperm(56, generator=gen)[:8]]
    rnd = torch.randint(0, k, (b, n_anchor - 16), generator=gen)
    A = anchors.to(hum)[torch.cat([near, mid, rnd.to(near)], 1)]
    return A


def test_warmup_draws_differ_per_rank_and_per_sample(monkeypatch):
    pred, targ = _stub()
    seen = []
    real = MX.path_mixture

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(out)
        return out
    monkeypatch.setattr(MX, "path_mixture", spy)
    out = {}
    for rank in (0, 1, 0):
        monkeypatch.setattr(H, "_world_rank", lambda r=rank: (2, r))
        out.setdefault(rank, []).append((H.warmup_paths(pred, targ, 24), seen[-1]))
    (p0, m0), (p0b, m0b), (p1, m1) = out[0][0], out[0][1], out[1][0]
    assert p0.shape == (4, 24, 8, 3)
    assert torch.equal(p0, p0b) and torch.equal(m0, m0b)                                      # same (iteration, rank)
    assert not torch.equal(p0, p1) and not torch.equal(m0[:, 8:16], m1[:, 8:16])              # rank 0 vs 1
    assert not all(torch.equal(m0[0, 8:16], m0[i, 8:16]) for i in range(1, 4))                 # mid per sample
    # the subset per sample: code each of the 80 slots by its index (no generator use in the stub)
    monkeypatch.setattr(MX, "path_mixture", lambda a, h, v, g: torch.arange(80.0)[None, :, None, None]
                        .expand(h.shape[0], 80, 8, 3).clone())
    monkeypatch.setattr(H, "_world_rank", lambda: (2, 0))
    sel = H.warmup_paths(pred, targ, 24)[..., 0, 0].long()
    assert all(len(set(r.tolist())) == 24 for r in sel)                                        # without replacement
    assert not all(torch.equal(sel[0], sel[i]) for i in range(1, 4))
    monkeypatch.setattr(H, "_world_rank", lambda: (2, 1))
    assert not torch.equal(H.warmup_paths(pred, targ, 24)[..., 0, 0].long(), sel)


def test_path_mixture_blocks_unchanged_except_the_mid_draw():
    pred, targ = _stub(same_hum=False)
    anc, hum = pred["trajectory_anchors"], targ["ex2_hum"]
    hv = torch.tensor([1.0, 1.0, 0.0, 1.0])
    new = MX.path_mixture(anc, hum, hv, torch.Generator().manual_seed(3))
    old = _old_path_mixture(anc, hum, hv, torch.Generator().manual_seed(3))
    assert new.shape == (4, 80, 8, 3)
    assert torch.equal(new[:, :8], old[:, :8])                                                 # near
    b = hum.shape[0]
    order = torch.cdist(hum.reshape(b, -1), anc.reshape(256, -1)).argsort(-1)
    for i in range(b):                                                                # mid: 8 distinct, ranks 8..63
        idx = [int((anc == new[i, j]).flatten(1).all(1).nonzero()[0]) for j in range(8, 16)]
        assert len(set(idx)) == 8 and set(idx) <= set(order[i, 8:64].tolist())
    ref = MX.path_mixture(anc, hum, hv, torch.Generator().manual_seed(4))
    assert torch.equal(ref[:, 32:64], new[:, 32:64])                                           # rule block (near only)
    assert torch.equal(ref[[0, 1, 3], 64:], new[[0, 1, 3], 64:])                               # GT block (human path)
    assert torch.equal(new[2, 64:], new[2, :16])                                      # no human path -> anchors


# ------------------------------------------------------------------------------------------------- F18 eval pick
def test_eval_pick_needs_plan_select_index():
    cfg = SimpleNamespace(embed_dims=32, v6c_hid=32, bev_h=10, bev_w=10, pc_range=[-32.0, 0.0, -1.0, 32.0, 64.0, 3.0],
                          num_navi_cmd=4)
    m = H.RefinerV6c(cfg)
    with pytest.raises(ValueError, match="plan_select_index"):
        m.pick({"cand_states": torch.zeros(2, 5, 41, 7)}, training=False)
    assert torch.equal(m.pick({"plan_select_index": torch.tensor([3, 1])}, training=False), torch.tensor([3, 1]))


# ------------------------------------------------------------------------------------------------- agent (F16, F17)
@pytest.fixture(scope="module")
def batch():
    if not have_inputs():
        pytest.skip("dataset / GT caches / teacher caches not on this host")
    if not FIX.is_file():
        make_fixture(FIX)
    return torch.load(FIX, map_location="cpu", weights_only=False)


def _agent_with_teachers(warmup):
    from navsim.agents.para_ssr.refiner_v6c.teacher import TeacherSet
    a = build_agent(0, v6c=True, v6c_stage="D", v6c_warmup_steps=warmup, v6c_teacher_ckpt="x")
    cfg = a._config
    torch.manual_seed(0)
    a._v6c_teacher_box[0] = TeacherSet(int(cfg.embed_dims), int(cfg.bev_h), int(cfg.bev_w)).eval()
    a.train()
    return a


def test_warmup_fast_path_is_the_mixture_loss(batch, monkeypatch):
    from navsim.agents.para_ssr.refiner_v6c.teacher import TeacherSet
    a = _agent_with_teachers(10)
    calls = []
    real = TeacherSet.fields
    monkeypatch.setattr(TeacherSet, "fields", lambda self, *x: calls.append(1) or real(self, *x))
    f, t = _fresh(batch)
    assert "kd_bev_0" in t and "cf_fold" in t
    p = a.forward(f, t)
    p["_v6c_iteration"] = 3
    v6c, ts = a.para_ssr_model.v6c, a._v6c_teachers(p["bev_embed"].device)
    loss, logs = H.v6c_loss(v6c, ts, a._config, f, t, p, True)
    assert len(calls) == 1                                                                     # warmup_risk_loss only
    assert float(logs["v6c/warmup"]) == 1.0 and "v6c/E_ovl_warm" in logs and "v6c/Eskill_ovl_warm" in logs
    assert "v6c/nobs_raised" in logs and "v6c/teacher_ok_frac_det" in logs and "v6c/teacher_ok_frac_map" in logs
    assert "v6c/E_ovl_base" not in logs
    ref = H.warmup_risk_loss(v6c, a._config, f, t, p, kd_teachers=ts)[0] + H.zero_touch(v6c)
    assert torch.equal(loss, ref)


def test_post_warmup_logs_present_and_step_exact(batch):
    a = _agent_with_teachers(0)
    f, t = _fresh(batch)
    p = a.forward(f, t)
    p["_v6c_iteration"] = 1
    assert torch.equal(p["v6c_lon_noop"], torch.ones(2))                                     # zero-initialised R-C
    loss, logs = H.v6c_loss(a.para_ssr_model.v6c, a._v6c_teachers(p["bev_embed"].device), a._config, f, t, p, True)
    assert torch.isfinite(loss) and float(logs["v6c/warmup"]) == 0.0
    keys = ["v6c/lon_noop", "v6c/lon_fixable", "v6c/lon_pass_frac", "v6c/z_lon_fix_sum", "v6c/z_lon_fix_tgt_sum",
            "v6c/z_lon_pass_sum", "v6c/ttc_overlabel_sum", "v6c/ttc_steppos", "v6c/ovl_overlabel_sum",
            "v6c/ovl_steppos"]
    keys += [f"v6c/Eskill_{nm}_{s}" for nm in ("ovl", "ttc", "offroad") for s in ("base", "v6")]
    keys += [f"v6c/KDcov_{nm}_{s}" for nm in N.E_NAMES for s in ("base", "v6")]
    for k in keys:
        assert k in logs and bool(torch.isfinite(logs[k])), k
    assert float(logs["v6c/lon_noop"]) == 1.0
    if "v6c/J_flip_frac" in logs:
        assert "v6c/J_pair_acc_sum" in logs
    for key, den in (("v6c/z_lon_fix", "v6c/lon_fixable"), ("v6c/z_lon_pass", "v6c/lon_pass_frac"),
                     ("v6c/ttc_overlabel", "v6c/ttc_steppos"), ("v6c/ovl_overlabel", "v6c/ovl_steppos"),
                     ("v6c/J_pair_acc", "v6c/J_flip_frac")):
        if key in logs:
            assert abs(float(logs[key + "_sum"]) / float(logs[den]) - float(logs[key])) < 1e-6, key
        elif den in logs:
            assert float(logs[den]) == 0.0 and float(logs[key + "_sum"]) == 0.0, key
