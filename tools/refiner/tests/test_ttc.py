"""Tests for the TTC surrogate term of stage-T run 3 (PRESTATED_DECISION_RULE AMENDMENT 4 (1)).  CPU, < 1 min.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests/test_ttc.py

- surrogate.ttc_cost: projection geometry vs a hand case (box straight ahead, constant speed: flagged iff v * delta brings
  it within the margin), object-time indexing (objects at n + 10 delta: a lead car at the ego's speed is never closer
  than now), beyond-horizon / not-OBS pairs excluded (not clamped), zero speed = no projection (== C_col exactly; delta 0
  == C_col for any speed, human mask included), gradient (braking lowers the cost; finite-difference check), the human
  mask from the human's PROJECTED boxes, details;
- defaults unchanged: DEFAULT_WEIGHTS / DEFAULT_W ttc = 0, m_ttc = 0, surrogate_loss bit-identical to the 5-term sum
  (even with a NaN TTC), the other terms independent of the TTC scene fields;
- train_refiner: --w ttc=..., --m-ttc, config.json records, resume refusal before the DONE exit, stub refusal, t_ttc
  step / epoch stats, scene_from_batch == scene_from_numpy for the TTC fields; stageT_gpu_commands.sh tag guard;
  m8_recheck.sub_scene subsets the TTC fields.
All data synthetic (no train / dev / navtest data read).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import surrogate as SU  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import decode  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import HALF_LEN, HALF_WID, RA2C, dense_reference  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a, **k: None  # noqa: E731
FRONT = RA2C + HALF_LEN                     # rear axle -> front bumper [m]
LN2T = SU.TEMP * math.log(2.0)              # smooth g of two equal axis gaps = gap - T ln 2 (others negligible)


def _dense(v, y=0.0, h=0.0, B=1, T=41):
    """constant-speed straight dense reference [B, T, 3] (x = v 0.1 n along heading h)."""
    n = torch.arange(T, dtype=torch.float64)
    s = v * 0.1 * n
    d = torch.stack([s * math.cos(h), y + s * math.sin(h), torch.full_like(s, h)], -1)
    return d[None].repeat(B, 1, 1)


def _scene(boxes_t, obs_t=None, agent=True, ttc=True, human=None):
    """one scene from per-time boxes [A, 51, 5] (objects at t = 0.1 m); ttc=False -> only the first 41 times."""
    bt = torch.as_tensor(np.asarray(boxes_t, np.float64)).reshape(-1, 51, 5)
    A = bt.shape[0]
    ot = torch.ones(A, 51, dtype=torch.bool) if obs_t is None else torch.as_tensor(obs_t).reshape(A, 51)
    kw = dict(boxes=bt[None, :, :41], obs=ot[None, :, :41],
              is_agent=torch.as_tensor(agent).reshape(-1).expand(A)[None].clone())
    if ttc:
        kw.update(boxes_ttc=bt[None], obs_ttc=ot[None])
    if human is not None:
        hd = torch.as_tensor(human, dtype=torch.float64).reshape(1, 41, 3)
        kw.update(human_dense=hd, human_overlap=SU.human_overlap_mask(hd, kw["boxes"]))
    return SU.SceneBatch(**kw)


def _static(x, y=0.0, h=0.0, L=4.0, W=2.0, A=1):
    return np.tile(np.array([x, y, h, L, W], np.float64), (A, 51, 1))


# ----------------------------------------------------------------------------------------------- geometry
def test_offsets_and_projection():
    assert SU.ttc_offsets(SU.TTC_DELTAS) == (3, 6, 9) and SU.ttc_offsets((0.0,)) == (0,)
    for bad in ((0.25,), (-0.1,)):
        with pytest.raises(ValueError):
            SU.ttc_offsets(bad)
    d = _dense(7.0, y=1.0, h=0.4)
    p = SU.project_poses(d, SU.TTC_DELTAS)
    v = SU.ref_speed(d)
    for k, dl in enumerate(SU.TTC_DELTAS):
        exp = d.clone()
        exp[..., 0] += v * dl * math.cos(0.4)
        exp[..., 1] += v * dl * math.sin(0.4)
        torch.testing.assert_close(p[:, :, k], exp, rtol=0, atol=1e-12)
    torch.testing.assert_close(v, torch.full_like(v, 7.0), rtol=0, atol=1e-12)
    still = _dense(0.0, y=2.0, h=-1.0)                                   # zero speed: projection exactly 0
    ps = SU.project_poses(still, SU.TTC_DELTAS)
    assert torch.equal(ps, still[:, :, None].expand_as(ps))


@pytest.mark.parametrize("v", [5.0, 12.0])
def test_hand_case_box_ahead(v):
    """Static box straight ahead, ego at constant speed v: the TTC gap at (n, k) is X - L/2 - FRONT - 0.1 v n - v delta_k;
    the minimum (n = 40, delta 0.9) is 0.9 v closer than the collision minimum (n = 40).  Flagged iff that is < m."""
    L, m = 4.0, 0.2
    base = L / 2 + FRONT + 4.0 * v
    for extra, want in ((0.9 * v - 0.05, True), (0.9 * v + 0.05, False), (0.9 * v - 0.5, True)):
        X = base + m + LN2T + extra                                      # smooth g = gap - T ln 2
        sc = _scene(_static(X, L=L))
        o = SU.ttc_cost(_dense(v), sc, cfg=SU.SurrogateConfig(m_ttc=m), details=True)
        c = SU.collision_cost(_dense(v), sc, cfg=SU.SurrogateConfig(m_col=m))
        assert float(o["gmin_hard"]) == pytest.approx(X - L / 2 - FRONT - 4.9 * v, abs=1e-9)
        assert float(c["gmin_hard"]) == pytest.approx(X - L / 2 - FRONT - 4.0 * v, abs=1e-9)
        assert float(o["gmin"]) == pytest.approx(float(o["gmin_hard"]) - LN2T, abs=1e-9)
        assert bool(o["flag"]) is want and bool(o["viol"] > 0) is want
        assert bool(c["viol"] > 0) is False                              # the unprojected collision term never flags
        if want:
            assert int(o["first_k"]) == 2 and int(o["first_obj"]) == 0 and 1 <= int(o["first_n"]) <= 40
        else:
            assert int(o["first_n"]) == -1 and int(o["first_k"]) == -1
        # flag_margin: any other margin from the same margin-free gmin
        o2 = SU.ttc_cost(_dense(v), sc, cfg=SU.SurrogateConfig(m_ttc=m), details=True, flag_margin=m + 1.0)
        assert bool(o2["flag"]) and torch.equal(o2["cost"], o["cost"])
    # per-pair values = the formula (all n, k)
    X = 80.0
    o = SU.ttc_cost(_dense(v), _scene(_static(X, L=L)), details=True)
    n = torch.arange(1, 41, dtype=torch.float64)[:, None]
    dl = torch.tensor(SU.TTC_DELTAS, dtype=torch.float64)[None]
    torch.testing.assert_close(o["g"][0, 0], X - L / 2 - FRONT - 0.1 * v * n - v * dl - LN2T, rtol=0, atol=1e-9)
    assert int(o["n_pairs"]) == 120


def test_object_time_index():
    """Objects are taken at dense time n + 10 delta: a lead car at the ego's own speed keeps a constant gap for every
    (n, delta) (if the objects were taken at n, the projection would close it by v delta)."""
    v, X, L = 10.0, 30.0, 4.0
    lead = _static(X, L=L)
    lead[0, :, 0] = X + v * 0.1 * np.arange(51)
    o = SU.ttc_cost(_dense(v), _scene(lead), details=True)
    torch.testing.assert_close(o["g"][0, 0], torch.full((40, 3), X - L / 2 - FRONT - LN2T, dtype=torch.float64),
                               rtol=0, atol=1e-9)
    # an object observed only at object time m0: counted exactly at (n, k) with n + 10 delta_k = m0
    m0 = 30
    obs = np.zeros((1, 51), bool)
    obs[0, m0] = True
    o = SU.ttc_cost(_dense(v), _scene(_static(200.0), obs_t=obs), details=True)
    got = {(int(i) + 1, int(k)) for i, k in o["mask"][0, 0].nonzero()}
    assert got == {(m0 - 3, 0), (m0 - 6, 1), (m0 - 9, 2)}


def test_beyond_horizon_and_not_obs_excluded():
    """Pairs whose object time is beyond the stored times, or where the object is not OBS (t > t_avail), are excluded
    -- never clamped to the last stored time."""
    v = 10.0
    near = _static(0.0)
    near[0, :, 0] = v * 0.1 * np.arange(51) + FRONT + 2.0 + 0.05       # 5 cm ahead of the bumper at every time
    full = SU.ttc_cost(_dense(v), _scene(near), details=True)
    assert int(full["n_pairs"]) == 120
    only41 = SU.ttc_cost(_dense(v), _scene(near, ttc=False), details=True)      # objects 0..4 s only
    assert int(only41["n_pairs"]) == 37 + 34 + 31
    assert float(only41["cost"]) < float(full["cost"])
    obs = np.ones((1, 51), bool)
    obs[0, 46:] = False                                                  # e.g. t_avail = 4.5 s
    part = SU.ttc_cost(_dense(v), _scene(near, obs_t=obs), details=True)
    assert int(part["n_pairs"]) == 40 + 39 + 36
    # an object that is on the ego only at object times 46..50 while NOT OBS there: cost 0 (clamping would count it)
    ghost = _static(300.0)
    ghost[0, 46:, 0] = v * 0.1 * np.arange(46, 51) + RA2C
    o = SU.ttc_cost(_dense(v), _scene(ghost, obs_t=obs), details=True)
    assert float(o["cost"]) < 1e-30 and float(o["gmin_hard"]) > 100
    o_obs = SU.ttc_cost(_dense(v), _scene(ghost), details=True)
    assert bool(o_obs["hard_overlap"]) and float(o_obs["cost"]) > 0.1
    # no objects -> zeros / -1
    empty = SU.SceneBatch(boxes=torch.zeros(1, 0, 41, 5, dtype=torch.float64), obs=torch.zeros(1, 0, 41, dtype=torch.bool),
                          is_agent=torch.zeros(1, 0, dtype=torch.bool))
    e = SU.ttc_cost(_dense(v, B=2), empty, details=True)
    assert torch.equal(e["cost"], torch.zeros(2, dtype=torch.float64)) and e["first_n"].tolist() == [-1, -1]
    assert not bool(e["flag"].any())


def test_zero_speed_and_delta_zero_equal_collision():
    rng = np.random.default_rng(3)
    A = 6
    bx = np.zeros((A, 51, 5))
    bx[:, :, 0] = rng.uniform(-3, 9, (A, 1))
    bx[:, :, 1] = rng.uniform(-3, 3, (A, 1))
    bx[:, :, 2] = rng.uniform(-1, 1, (A, 1))
    bx[:, :, 3], bx[:, :, 4] = 4.0, 1.8
    agent = torch.as_tensor(rng.random(A) > 0.5)
    still = _dense(0.0, y=0.2, h=0.1, B=2)
    m = 0.3
    for deltas, fac in (((0.3,), 1), ((0.3, 0.6, 0.9), 3)):
        cfg = SU.SurrogateConfig(m_col=m, m_ttc=m, ttc_deltas=deltas)
        sc = _scene(bx, agent=agent)
        t = SU.ttc_cost(still, sc, cfg=cfg, details=True)
        c = SU.collision_cost(still, sc, cfg=cfg)
        torch.testing.assert_close(t["cost"], fac * c["cost"], rtol=1e-12, atol=0)
        assert torch.equal(t["gmin"], c["gmin"]) and float(c["cost"][0]) > 0
    # delta 0 is C_col for any speed / moving objects, human mask included
    mov = bx.copy()
    mov[:, :, 0] += 0.4 * np.arange(51)[None]
    human = _dense(9.0, y=0.3)[0].numpy()
    sc = _scene(mov, agent=agent, human=human)
    cfg = SU.SurrogateConfig(m_col=m, m_ttc=m, ttc_deltas=(0.0,))
    d = _dense(8.0, y=0.1, B=3)
    t = SU.ttc_cost(d, sc, cfg=cfg, details=True)
    c = SU.collision_cost(d, sc, cfg=cfg)
    torch.testing.assert_close(t["cost"], c["cost"], rtol=1e-12, atol=0)
    assert torch.equal(t["mask"][..., 0], c["mask"])
    assert int(c["n_human_masked"].sum()) > 0 and torch.equal(t["n_human_masked"], c["n_human_masked"])
    hm = SU.human_ttc_overlap_mask(sc.human_dense, sc.boxes_ttc, deltas=(0.0,))
    assert torch.equal(hm[..., 0], sc.human_overlap[:, :, 1:])


def test_gradient_braking_lowers_cost():
    """Static box ahead, ego speed s * v: C_ttc decreases when braking (s < 1) and dC/ds > 0 (descent brakes); the
    gradient matches a finite difference.  Also dC/dv through the projection alone (positions fixed)."""
    v, L = 10.0, 4.0
    X = L / 2 + FRONT + 4.9 * v - 0.3
    sc = _scene(_static(X, L=L))
    cfg = SU.SurrogateConfig(m_ttc=0.1)
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    base = _dense(v)
    d = base * torch.stack([s, s, torch.ones_like(s)])
    c = SU.ttc_cost(d, sc, cfg=cfg)["cost"].sum()
    c.backward()
    assert float(s.grad) > 0
    f = lambda x: float(SU.ttc_cost(base * torch.tensor([x, x, 1.0], dtype=torch.float64), sc, cfg=cfg)["cost"])  # noqa
    assert f(0.9) < f(1.0) and float(s.grad) == pytest.approx((f(1 + 1e-6) - f(1 - 1e-6)) / 2e-6, rel=1e-4)
    # the collision term alone does not see this box (it is beyond the 4 s reach): the gradient comes from the speed
    assert float(SU.collision_cost(base, sc, cfg=SU.SurrogateConfig(m_col=0.1))["cost"]) < 1e-9
    # speed only: same pose at n = 20, different speed -> different projected cost
    obs = np.zeros((1, 51), bool)
    obs[0, 29] = True                                                    # object seen only at n + 9 = 29 (n = 20)
    X2 = 2.0 * v + L / 2 + FRONT + 0.9 * v - 0.2                         # projected (0.9 s) bumper 0.2 m inside
    sc2 = _scene(_static(X2, L=L), obs_t=obs)
    d2 = torch.zeros(1, 41, 3, dtype=torch.float64)
    vv = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    d2 = d2 + torch.stack([torch.cat([2.0 * v + vv * 0.1 * (torch.arange(41.0, dtype=torch.float64) - 20)]),
                           torch.zeros(41, dtype=torch.float64), torch.zeros(41, dtype=torch.float64)], -1)[None]
    cfg2 = SU.SurrogateConfig(m_ttc=0.1, ttc_deltas=(0.9,))              # only the pair (n = 20, delta 0.9)
    assert int(SU.ttc_cost(d2.detach(), sc2, cfg=cfg2, details=True)["n_pairs"]) == 1
    c2 = SU.ttc_cost(d2, sc2, cfg=cfg2)["cost"].sum()
    c2.backward()
    assert float(vv.grad) > 0                                            # d x_20 / d v = 0: only the projection


def test_human_mask_uses_projected_human_boxes():
    """The human's own PROJECTED boxes define the TTC mask: a human that stops 0.5 m short of a static box (so its
    unprojected footprint never overlaps -> no C_col mask) but whose projection does overlap masks those (j, n, k)."""
    L = 4.0
    X = 30.0
    # human: 10 m/s, then stops with the bumper 0.5 m short of the box (dense 0.1 s), heading 0
    x_stop = X - L / 2 - FRONT - 0.5
    xs, x, vcur = [], 0.0, 10.0
    for n in range(41):
        xs.append(x)
        x = min(x + vcur * 0.1, x_stop)
    hd = np.stack([np.array(xs), np.zeros(41), np.zeros(41)], -1)
    sc = _scene(_static(X, L=L), human=hd)
    assert not bool(sc.human_overlap.any())                             # unprojected: never overlaps
    hm = SU.human_ttc_overlap_mask(sc.human_dense, sc.boxes_ttc)
    assert bool(hm.any())
    # exact recompute
    hp = SU.project_poses(sc.human_dense, SU.TTC_DELTAS)[:, None, 1:]
    tix = torch.arange(1, 41)[:, None] + torch.tensor([3, 6, 9])
    exp = SU.box_gaps(hp, sc.boxes_ttc[:, :, tix]).max(-1).values <= 0
    assert torch.equal(hm, exp)
    draft = torch.as_tensor(hd)[None]
    on = SU.ttc_cost(draft, sc, details=True)
    off = SU.ttc_cost(draft, sc, cfg=SU.SurrogateConfig(use_human_mask=False), details=True)
    assert int(on["n_human_masked"]) == int((hm & off["mask"] & (off["g"] < 0)).sum()) > 0
    assert torch.equal(on["mask"], off["mask"] & ~hm) and float(on["cost"]) < float(off["cost"])
    # another draft driving through the box is still penalised where the human's projection did not overlap
    fast = SU.ttc_cost(_dense(12.0), sc, details=True)
    assert bool(fast["hard_overlap"]) and float(fast["cost"]) > 0
    # human far to the left: no TTC mask
    sc2 = _scene(_static(X, L=L), human=hd + np.array([0.0, 20.0, 0.0]))
    assert not bool(SU.human_ttc_overlap_mask(sc2.human_dense, sc2.boxes_ttc).any())


def test_not_behind_uses_unprojected_pose():
    """Not-behind mask: object centre at n + k vs the UNPROJECTED ego centre / heading at n (official is_agent_behind
    uses the rear axle at time_idx)."""
    v = 10.0
    b = _static(0.0)
    # object at object time m sits 1.0 m behind the unprojected ego centre at m - 3 (lon = -1.0 >= -1.3 -> counted for
    # delta 0.3), i.e. 4.0 m behind the ego centre at its own time m (a mask on the ego at time m would drop it)
    b[0, :, 0] = v * 0.1 * (np.arange(51) - 3) + RA2C - 1.0
    o = SU.ttc_cost(_dense(v), _scene(b), details=True)
    assert bool(o["mask"][0, 0, :, 0].all())
    b[0, :, 0] -= 0.5                                                    # lon = -1.5 < -1.3 -> behind
    o = SU.ttc_cost(_dense(v), _scene(b), details=True)
    assert not bool(o["mask"][0, 0, :, 0].any())
    o = SU.ttc_cost(_dense(v), _scene(b), cfg=SU.SurrogateConfig(behind_m=None), details=True)
    assert bool(o["mask"][0, 0, :, 0].all())


def test_min_speed_option_and_weights():
    v, L = 10.0, 4.0
    X = L / 2 + FRONT + 4.9 * v - 0.5
    sc = _scene(np.concatenate([_static(X, L=L), _static(X, L=L)]), agent=torch.tensor([True, False]))
    o = SU.ttc_cost(_dense(v), sc, details=True)
    pen = SU.BETA * torch.nn.functional.softplus((0.0 - o["g"]) / SU.BETA) * o["mask"]
    exp = (pen[0, 0] * 1.0 + pen[0, 1] * 0.5).sum(-1).mean()
    torch.testing.assert_close(o["cost"][0], exp, rtol=1e-12, atol=0)
    assert float(o["cost"]) > 0
    assert float(SU.ttc_cost(_dense(v), sc, cfg=SU.SurrogateConfig(ttc_min_speed=5e-3))["cost"]) == float(o["cost"])
    close = _scene(_static(L / 2 + FRONT + 0.05, L=L))                  # stopped 5 cm behind a box
    assert float(SU.ttc_cost(_dense(0.0), close)["cost"]) > 0            # default: no stopped-ego skip
    assert float(SU.ttc_cost(_dense(0.0), close, cfg=SU.SurrogateConfig(ttc_min_speed=5e-3))["cost"]) == 0.0


# ----------------------------------------------------------------------------------------------- defaults
def _dec_and_scene():
    torch.manual_seed(0)
    tau = torch.as_tensor(np.stack([SY.straight(8.0 + i, 0.1 * i) for i in range(4)]), dtype=torch.float64)
    z = torch.randn(4, 6, dtype=torch.float64) * 0.3
    w = torch.randn(4, 6, dtype=torch.float64) * 0.3
    dec = decode(tau, z, w, torch.full((4,), 8.0, dtype=torch.float64))
    bx = _static(33.0, y=0.4)
    mv = _static(10.0, y=3.0)
    mv[0, :, 0] += 0.5 * np.arange(51)
    human = dense_reference(tau[:1])[0].numpy()
    return dec, tau, _scene(np.concatenate([bx, mv]), agent=torch.tensor([False, True]), human=human)


def test_defaults_unchanged():
    assert SU.DEFAULT_WEIGHTS["ttc"] == 0.0 and TR.DEFAULT_W["ttc"] == 0.0 and SU.SurrogateConfig().m_ttc == 0.0
    assert {k: v for k, v in SU.DEFAULT_WEIGHTS.items() if k != "ttc"} == \
        {"col": 1.0, "dac": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1, "gate": 0.5}
    assert {k: v for k, v in TR.DEFAULT_W.items() if k != "ttc"} == \
        {"col": 1.0, "dac": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1}
    assert TR.get_parser().parse_args(["--arm", "none"]).m_ttc is None
    assert TR.surrogate_config(TR.get_parser().parse_args(["--arm", "none"])) is None
    dec, tau, sc = _dec_and_scene()
    total, t = SU.surrogate_loss(dec, tau, sc)
    assert float(t["ttc"].max()) > 0                                     # computed (and logged), weight 0
    per = sum(SU.DEFAULT_WEIGHTS[k] * t[k] for k in ("col", "dac", "prog", "cmf", "mod"))
    assert torch.equal(t["per_draft"], per) and torch.equal(total, per.mean())
    # the TTC scene fields do not touch the other terms
    import dataclasses
    bare = dataclasses.replace(sc, boxes_ttc=None, obs_ttc=None, human_dense=None)
    t0 = SU.surrogate_terms(dec, tau, bare)
    t1 = SU.surrogate_terms(dec, tau, sc)
    for k in ("col", "dac", "prog", "cmf", "mod", "P1", "P0", "unknown"):
        assert torch.equal(t0[k], t1[k]), k
    # enabled: + w * ttc exactly
    W = dict(SU.DEFAULT_WEIGHTS, ttc=1.0)
    _, t2 = SU.surrogate_loss(dec, tau, sc, weights=W)
    torch.testing.assert_close(t2["per_draft"], per + t["ttc"], rtol=0, atol=0)


def test_default_loss_ignores_nonfinite_ttc(monkeypatch):
    dec, tau, sc = _dec_and_scene()
    total, t = SU.surrogate_loss(dec, tau, sc)
    real = SU.ttc_cost

    def nan_ttc(*a, **k):
        o = real(*a, **k)
        o["cost"] = torch.full_like(o["cost"], float("nan"))
        return o
    monkeypatch.setattr(SU, "ttc_cost", nan_ttc)
    total2, t2 = SU.surrogate_loss(dec, tau, sc)
    assert torch.equal(total, total2) and torch.isnan(t2["ttc"]).all()


def test_ttc_no_graph_at_weight_zero():
    """Reviewer fix: TTC weight 0 -> the TTC term is computed without autograd (same values, no graph); weight > 0 ->
    it carries a gradient.  The other terms keep theirs."""
    torch.manual_seed(0)
    tau = torch.as_tensor(np.stack([SY.straight(8.0 + i, 0.1 * i) for i in range(4)]), dtype=torch.float64)
    z = (torch.randn(4, 6, dtype=torch.float64) * 0.3).requires_grad_(True)
    w = torch.zeros(4, 6, dtype=torch.float64)
    dec = decode(tau, z, w, torch.full((4,), 8.0, dtype=torch.float64))
    _, _, sc = _dec_and_scene()
    cfg = SU.SurrogateConfig(m_ttc=0.3)
    t_on = SU.surrogate_terms(dec, tau, sc, cfg=cfg)
    t_off = SU.surrogate_terms(dec, tau, sc, cfg=cfg, ttc_grad=False)
    assert t_on["ttc"].requires_grad and not t_off["ttc"].requires_grad and t_off["col"].requires_grad
    assert float(t_on["ttc"].detach().max()) > 0
    for k in t_on:
        assert torch.equal(t_on[k].detach(), t_off[k].detach()), k
    _, t0 = SU.surrogate_loss(dec, tau, sc, cfg=cfg)
    _, t1 = SU.surrogate_loss(dec, tau, sc, cfg=cfg, weights=dict(SU.DEFAULT_WEIGHTS, ttc=1.0))
    assert not t0["ttc"].requires_grad and t1["ttc"].requires_grad
    g, = torch.autograd.grad(t1["ttc"].sum(), z)
    assert float(g.abs().sum()) > 0


# ----------------------------------------------------------------------------------------------- train_refiner
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ttc")
    df, src = SY.make_sources(tmp, 10)
    RD.pack_split("train", df, tmp / "packed", sources=src, workers=1, log_fn=QUIET)
    teacher = SY.make_fake_teacher(tmp, df.token)
    return tmp, df, teacher


def _args(tmp, teacher, extra=(), sur="stub"):
    a = ["--arm", "none", "--fold", "0", "--seed", "0", "--gpu", "-1", "--packed-root", str(tmp / "packed"),
         "--runs", str(tmp / "runs"), "--teacher-root", str(teacher), "--tokens-per-batch", "3", "--workers", "0",
         "--n-norm", "4", "--log-every", "1", "--inner-val-frac", "0.3", "--max-steps", "1", "--epochs", "1",
         "--max-val-batches", "1", "--surrogate", sur, *extra]
    return TR.get_parser().parse_args(a)


def test_parse_weights_and_config():
    assert TR.parse_weights("col=1,ttc=1")["ttc"] == 1.0 and TR.parse_weights("")["ttc"] == 0.0
    with pytest.raises(SystemExit, match="unknown term"):
        TR.parse_weights("col=1,tcc=1")
    a = TR.get_parser().parse_args(["--arm", "none", "--m-ttc", "0.1", "--m-col", "0.15"])
    cfg = TR.surrogate_config(a)
    assert cfg.m_ttc == 0.1 and cfg.m_col == 0.15 and cfg.m_dac == SU.M_DAC and cfg.ttc_deltas == SU.TTC_DELTAS


def test_scene_from_batch_ttc_fields(env):
    tmp, df, _ = env
    from navsim.agents.para_ssr.refiner.gt_future import load_objects
    P = RD.PackedSplit("train", tmp / "packed")
    rows = [3, 8, 9]
    b = RD.collate_tokens([RD.TokenDataset(P, rows, None)[i] for i in range(len(rows))])
    got = TR.scene_from_batch(b, torch.float64)
    ref = SU.collate_scenes([SU.scene_from_numpy(objs=load_objects(tmp / "objects" / "train" / f"{df.token[r]}.npz"),
                                                 human_traj=P.row(r)["human_traj"]) for r in rows])
    for f in ("boxes_ttc", "obs_ttc", "human_dense"):
        x, y = getattr(got, f), getattr(ref, f)
        assert x.shape == y.shape, f
        if x.dtype == torch.bool:
            assert torch.equal(x, y), f
        else:
            torch.testing.assert_close(x.double(), y.double(), atol=1e-5, rtol=0, msg=f)
    assert got.boxes_ttc.shape[2] == 51 and torch.equal(got.obs_ttc[:, :, :41], got.obs)
    torch.testing.assert_close(got.boxes_ttc[:, :, :41], got.boxes, rtol=0, atol=0)
    T, K = b["tau0"].shape[:2]
    dec = decode(b["tau0"].reshape(-1, 8, 3).double(), torch.zeros(T * K, 6, dtype=torch.float64),
                 torch.zeros(T * K, 6, dtype=torch.float64), v0=b["v0"].double().repeat_interleave(K))
    tidx = torch.arange(T).repeat_interleave(K)
    cfg = SU.SurrogateConfig(m_ttc=0.5)
    t1 = SU.ttc_cost(dec["dense"], ref, tidx, cfg)["cost"]
    t2 = TR.surrogate_terms_batch(dec, b, T, K, cfg)["ttc"]
    torch.testing.assert_close(t2, t1, atol=1e-6, rtol=1e-6)
    assert float(t2.max()) > 1e-3                                        # the cone at x = 25 m is reached


def test_compute_loss_ttc_weight(env):
    tmp = env[0]
    P = RD.PackedSplit("train", tmp / "packed")
    b = RD.collate_tokens([RD.TokenDataset(P, [8, 9], None)[i] for i in range(2)])
    T, K = b["tau0"].shape[:2]
    out = {"z_lon": torch.zeros(T, K, 6), "w_lat": torch.zeros(T, K, 6), "gate_logit": torch.zeros(T, K)}
    cfg = SU.SurrogateConfig(m_ttc=0.3)
    l0, s0, _ = TR.compute_loss(out, b, TR.surrogate_terms_batch, dict(TR.DEFAULT_W), 1.0, cfg=cfg)
    l1, s1, _ = TR.compute_loss(out, b, TR.surrogate_terms_batch, dict(TR.DEFAULT_W, ttc=2.0), 1.0, cfg=cfg)
    assert s0["t_ttc"] == s1["t_ttc"] > 0
    seen = {}
    fn = TR.surrogate_terms_batch

    def spy(*a, **k):
        seen.setdefault("ttc_grad", []).append(k.get("ttc_grad"))
        return fn(*a, **k)
    TR.compute_loss(out, b, spy, dict(TR.DEFAULT_W), 1.0, cfg=cfg)
    TR.compute_loss(out, b, spy, dict(TR.DEFAULT_W, ttc=2.0), 1.0, cfg=cfg)
    assert seen["ttc_grad"] == [False, True]                             # w_ttc 0: logged only, no graph
    assert float(l1) == pytest.approx(float(l0) + 2.0 * s0["t_ttc"], rel=1e-6)
    # the default loss is the 5-term loss (t_ttc only logged)
    exp = sum(TR.DEFAULT_W[k] * s0[f"t_{k}"] for k in ("col", "dac", "prog", "cmf", "mod")) + s0["gate_bce"]
    assert float(l0) == pytest.approx(exp, rel=1e-6)


def test_train_records_and_logs(env):
    tmp, _, teacher = env
    run = TR.train(_args(tmp, teacher, ["--m-ttc", "0.1", "--w", "ttc=1", "--tag", "ttc"], sur="real"))
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["m_ttc"] == 0.1 and cfg["weights"]["ttc"] == 1.0 and cfg["surrogate_cfg"]["m_ttc"] == 0.1
    assert list(cfg["surrogate_cfg"]["ttc_deltas"]) == [0.3, 0.6, 0.9]
    log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    st = [r for r in log if r["kind"] == "step"][0]
    ep = [r for r in log if r["kind"] == "epoch"][0]
    assert "t_ttc" in st and "t_ttc" in ep["train"] and "t_ttc" in ep["val"]
    # default call records ttc weight 0 and m_ttc None
    run0 = TR.train(_args(tmp, teacher, ["--tag", "ttc0"]))
    c0 = json.loads((run0 / "config.json").read_text())
    assert c0["m_ttc"] is None and c0["weights"]["ttc"] == 0.0


def test_resume_refusal(env):
    tmp, _, teacher = env
    TR.train(_args(tmp, teacher, ["--m-ttc", "0.1", "--w", "ttc=1", "--tag", "rr"], sur="real"))
    for extra in (["--m-ttc", "0.05", "--w", "ttc=1"], ["--m-ttc", "0.1"], ["--w", "ttc=1"], []):
        with pytest.raises(SystemExit, match="config.json has"):
            TR.train(_args(tmp, teacher, [*extra, "--tag", "rr"], sur="real"))
    TR.train(_args(tmp, teacher, ["--m-ttc", "0.1", "--w", "ttc=1", "--tag", "rr"], sur="real"))     # same: resumes


def test_finished_run1_run2_style_refuses_ttc(env):
    """A finished run-2 style directory (config.json without m_ttc / weights['ttc']): the default call exits cleanly
    through DONE; any TTC option raises SystemExit BEFORE the DONE exit, and nothing in the directory changes."""
    tmp, _, teacher = env
    run = TR.train(_args(tmp, teacher, ["--tag", "r2"]))
    c = json.loads((run / "config.json").read_text())
    c.pop("m_ttc")
    c["weights"].pop("ttc")
    (run / "config.json").write_text(json.dumps(c))
    (run / "DONE").write_text("x")
    before = {f.name: f.read_bytes() for f in run.iterdir() if f.is_file()}
    assert TR.train(_args(tmp, teacher, ["--tag", "r2"])) == run
    assert TR.train(_args(tmp, teacher, ["--tag", "r2", "--w", "ttc=0"])) == run
    for extra in (["--m-ttc", "0.0"], ["--w", "col=1,ttc=1"], ["--m-ttc", "0.1", "--w", "ttc=1"]):
        with pytest.raises(SystemExit, match="config.json has"):
            TR.train(_args(tmp, teacher, [*extra, "--tag", "r2"], sur="real"))
    assert {f.name: f.read_bytes() for f in run.iterdir() if f.is_file()} == before


@pytest.mark.parametrize("extra,sur", [(["--w", "ttc=1"], "stub"), (["--m-ttc", "0.1"], "stub"),
                                       (["--m-ttc", "nan"], "real"), (["--w", "ttc=-1"], "real"),
                                       (["--w", "ttc=inf"], "real"), (["--w", "tcc=1"], "real")])
def test_bad_ttc_args_refused(env, extra, sur):
    tmp, _, teacher = env
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, [*extra, "--tag", "bad_ttc"], sur=sur))
    assert not (tmp / "runs" / "bad_ttc_none_fold0_seed0" / "config.json").exists()


def test_gpu_script_refuses_ttc_with_old_tags(tmp_path):
    src = (Path(TR.__file__).resolve().parent / "stageT_gpu_commands.sh").read_text()
    rec = tmp_path / "called"
    py = tmp_path / "py.sh"
    py.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {rec}\n")
    py.chmod(0o755)
    lines = []
    for ln in src.splitlines():
        if ln.startswith("DATA="):
            ln = f"DATA={tmp_path / 'data'}"
        elif ln.startswith("PY="):
            ln = f"PY={py}"
        lines.append(ln)
    assert sum(x.startswith(("DATA=", "PY=")) for x in src.splitlines()) == 2
    sh = tmp_path / "gpu.sh"
    sh.write_text("\n".join(lines) + "\n")
    env = {k: v for k, v in os.environ.items() if k != "TAG"}
    for tag in (None, "stageT", "stageT2"):
        e = env if tag is None else {**env, "TAG": tag}
        for extra in (["--m-ttc", "0.1"], ["--m-ttc=0.1"], ["--w", "col=1,ttc=1"], ["--w=ttc=1"]):
            r = subprocess.run(["bash", str(sh), "0", "none", "0", "0", *extra], env=e, capture_output=True, text=True)
            assert r.returncode == 2 and "refused" in r.stderr, (tag, extra)
    assert not rec.exists() and not (tmp_path / "data").exists()
    r = subprocess.run(["bash", str(sh), "0", "none", "0", "0", "--m-ttc", "0.1", "--w", "ttc=1"],
                       env={**env, "TAG": "stageT3"}, capture_output=True, text=True)
    assert r.returncode == 0 and "--tag stageT3 --m-ttc 0.1 --w ttc=1" in rec.read_text()


def test_m8_recheck_sub_scene_subsets_ttc_fields():
    import m8_recheck as R
    sc = _scene(np.concatenate([_static(10.0), _static(20.0), _static(30.0)]), human=_dense(5.0)[0].numpy())
    keep = torch.tensor([True, False, True])
    sub = R.sub_scene(sc, keep)
    assert sub.boxes_ttc.shape[1] == 2 and torch.equal(sub.boxes_ttc, sc.boxes_ttc[:, [0, 2]])
    assert torch.equal(sub.obs_ttc, sc.obs_ttc[:, [0, 2]]) and torch.equal(sub.human_dense, sc.human_dense)
    d = _dense(9.0, B=2)
    torch.testing.assert_close(SU.ttc_cost(d, sub)["cost"],
                               SU.ttc_cost(d, sc, cfg=SU.SurrogateConfig())["cost"] -
                               SU.ttc_cost(d, _scene(_static(20.0), human=_dense(5.0)[0].numpy()))["cost"],
                               rtol=1e-9, atol=1e-12)


def test_ttc_flag_check_confusion():
    import pandas as pd
    import ttc_flag_check as FC
    c = FC.confusion([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1) and c["recall"] == pytest.approx(2 / 3)
    df = pd.DataFrame(dict(token=["a"] * 3 + ["b"] * 2, k=[0, 1, 2, 0, 1], ttc=[1.0, 0.0, 1.0, 1.0, 0.0],
                           col_g=[1.0, 0.1, 1.0, 1.0, 1.0], ttc_g=[0.5, 0.5, -0.1, 0.2, -0.3],
                           ttc_g_stop=[0.5, 0.5, 0.5, 0.2, -0.3]))
    s = FC.summarize(df, 0.0, 0.15)
    assert s["proj"]["bank"]["tp"] == 1 and s["proj"]["bank"]["fp"] == 1 and s["combo"]["bank"]["tp"] == 2
    assert s["proj_stopskip"]["bank"]["fp"] == 0 and s["combo"]["human_k0"]["n"] == 2
