"""refiner-v3 temporal branch model tests (CPU, synthetic; docs/refiner_v3/SPEC.md sections 4.1, 4.3-4.6).

  config     : TemporalCfg round trip / validation / privileged flag
  forecast   : trim_objects; ca_forecast == a3 Worker.history + world('CA') (statics held) in float64; float32 close;
               objects.ca_forecast_np when B1's module exists; oracle rules; LearnedForecaster shapes / statics /
               mode diversity / no NaN (zero objects, one object, invalid past frames) / slot-permutation
               equivariance / trim invariance / determinism
  raster     : known answers (static on the path: splat weights, P_lat -> 1, inc / ext; lateral Laplace overlap; moving
               vehicle vclose sign; ego cov; taps gated after 4 s; known mask of a 40 m candidate), variants with
               identical stations share agent channels (ego channels differ), zero objects, raster gradient reaches mu
  gap        : gap_scalars == the a3 SAT + 9-feature reduction (a3 source executed) on synthetic scenes for the S world
               and the forecast world (M = 1); M = 2 == independent numpy reference of the per-object definition
  loss       : known answer (M = 2 toy), zero-target batch (finite 0, gradients present), NaN targets ignored
  branch     : shared init identical in arms a / b, param counts (237,888 shared), zero-init == no-op, gradient paths
               (fc_backprop False: critic never reaches the forecaster; True: it does), every parameter finite grad,
               causality guards (targets never forward, future-only data cannot change outputs, age checks),
               privileged-key rules, candidate permutation, determinism; trunk integration via the SPEC 4.5 hook
               (ON at init == OFF bitwise; gradients reach trunk + branch)
  speed      : CPU ms per token (scene) / per candidate (features), recorded in the test log (-s)

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_v3_temporal_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CK = str(Path(__file__).resolve().parents[1])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import forecast as FC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import raster as RA  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.branch import (TemporalBranch, objs_from_batch, trim_forecast)  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.loss import forecast_loss  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_temporal_testlib as TL  # noqa: E402

HAVE_A3 = TL.A3_PATH.is_file()
CA_B = torch.tensor([[0.2 + 0.15 * k for k in range(10)], [0.15 + 0.1 * k for k in range(10)]])


@pytest.fixture(scope="module")
def A3():
    if not HAVE_A3:
        pytest.skip("a3_conflict_features.py not available")
    return TL.load_a3()


def cfg_of(arm: str, **kw) -> TC.TemporalCfg:
    """arm a: fc_train 'joint' unless given (these tests exercise the trainable forecaster; the frozen Step-1 default
    has its own tests)."""
    f = {"a": "learned", "b": "ca", "d": "oracle_gt"}[arm]
    if f == "learned":
        kw.setdefault("fc_train", "joint")
    return TC.TemporalCfg(on=True, forecaster=f, n_modes=6 if f == "learned" else 1, **kw)


def branch(arm: str, seed: int = 0, **kw) -> TemporalBranch:
    return TemporalBranch(cfg_of(arm, **kw), 192, seed, ca_b=None if arm == "a" else CA_B)


def objs_dict(obj, mask, fv):
    return {"obj_in": obj, "obj_mask": mask, "frame_valid": fv}


def fc_static(c0, h0=0.0, L=1.0, W=1.0, group=2, vel=(0.0, 0.0), b=None, T=1):
    """Single-object M = 1 Forecast moving at constant velocity `vel` (knots exact)."""
    c0 = torch.tensor([[c0]], dtype=torch.float32).reshape(T, 1, 2)
    v = torch.tensor(vel, dtype=torch.float32).reshape(1, 1, 2)
    t = torch.tensor(TC.FC_T)
    mu = (c0[:, :, None, :] + v[:, :, None, :] * t[None, None, :, None])[:, :, None]   # [T,1,1,10,2]
    bb = torch.full_like(mu, TC.B0 if b is None else b)
    return FC.Forecast(mu=mu, b=bb, pi=torch.ones(T, 1, 1), present=torch.ones(T, 1, 1, 10), c0=c0,
                       v0=v.expand(T, 1, 2), h0=torch.full((T, 1), float(h0)), L=torch.full((T, 1), float(L)),
                       W=torch.full((T, 1), float(W)), group=torch.full((T, 1), group, dtype=torch.long),
                       valid=torch.ones(T, 1, dtype=torch.bool))


# ----------------------------------------------------------------------------------------------- config
def test_config_round_trip_and_validation():
    off = TC.TemporalCfg()
    assert not off.on and not off.privileged and TC.TemporalCfg.from_dict(None) == off
    assert TC.TemporalCfg.from_dict({}) == off
    for arm in "abd":
        c = cfg_of(arm)
        c.validate("T")
        d = c.to_dict()
        assert d["privileged"] == (arm == "d") and TC.TemporalCfg.from_dict(d) == c
    with pytest.raises(ValueError):
        TC.TemporalCfg.from_dict({"on": True, "bogus": 1})
    with pytest.raises(ValueError):
        cfg_of("a").validate("M")
    with pytest.raises(NotImplementedError):
        TC.TemporalCfg(on=True, source="v2det").validate("T")
    with pytest.raises(ValueError):
        TC.TemporalCfg(on=True, forecaster="ca", n_modes=6).validate("T")
    with pytest.raises(ValueError):
        TC.TemporalCfg(on=True, forecaster="learned", n_modes=9).validate("T")
    assert TC.TemporalCfg(on=True, source="gt").privileged
    TC.TemporalCfg().validate("M")                                             # off: always fine
    assert abs(TC.EGO_REAR + 1.127) < 1e-12 and abs(TC.EGO_FRONT - 4.049) < 1e-12
    assert len(TC.R_T) == TC.N_RT == 21 and TC.FC_T[-1] == 5.0 and len(RA.RASTER_CHANNELS) == 18


# ----------------------------------------------------------------------------------------------- trim / CA
def test_trim_objects():
    (obj, mask, fv), _ = TL.rand_objs(2, (5, 3), seed=1)
    K = int(torch.nonzero(mask.reshape(-1, TC.K_OBJ).any(0)).max()) + 1
    o, m, f = FC.trim_objects(obj, mask, fv)
    assert o.shape[2] == m.shape[2] == K and torch.equal(o, obj[:, :, :K]) and f is fv
    o, m, f = FC.trim_objects(obj, torch.zeros_like(mask), fv)
    assert o.shape[2] == 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ca_forecast_equals_a3_history_and_world(A3, seed):
    g = np.random.default_rng(seed)
    toks = [TL.rand_token(g, n) for n in (14, 9, 1)]
    toks.append([toks[0][0], np.zeros((0, 17)), np.zeros((0, 17))])          # no history at all
    obj, mask, fv = TL.pack_tokens(toks, dtype=torch.float64)
    fc = FC.ca_forecast(obj, mask, fv, CA_B.double())
    fc.check()
    t = np.asarray(TC.FC_T)
    n_hist = 0
    for i, frames in enumerate(toks):
        b0, l0 = TL.rows_to_ft(frames[0])
        acc, has, _ = A3.Worker.history(TL.FakeHistory(frames), "t0", b0, l0)
        n = len(b0)
        np.testing.assert_array_equal(fc.aux["has"][i, :n].numpy(), has)
        np.testing.assert_allclose(fc.aux["acc"][i, :n].numpy(), acc, rtol=0, atol=1e-9)
        c, *_ = A3.Worker.world(None, "CA", t, b0, l0, acc, None)            # [10, N, 2]
        ag = np.isin(l0, A3.AGENT)
        mu = fc.mu[i, :n, 0].numpy()                                          # [N, 10, 2]
        np.testing.assert_allclose(mu[ag], np.transpose(c, (1, 0, 2))[ag], rtol=0, atol=1e-9)
        np.testing.assert_array_equal(mu[~ag], np.broadcast_to(b0[~ag, None, :2], mu[~ag].shape))   # statics held
        n_hist += int(has.sum())
    assert n_hist > 5
    # float32 (model precision) stays close
    fc32 = FC.ca_forecast(obj.float(), mask, fv, CA_B)
    np.testing.assert_allclose(fc32.mu.double().numpy(), fc.mu.numpy(), rtol=0, atol=2e-4)
    # b: CA scales by group, B0 for statics / padding
    grp = fc.group[0, :14]
    assert torch.allclose(fc.b[0, :14, 0, :, 0][grp == 0], CA_B.double()[0].expand(int((grp == 0).sum()), 10))
    assert torch.all(fc.b[0, 14:] == TC.B0) and torch.all(fc.pi == 1)


def test_ca_forecast_equals_objects_ca_forecast_np():
    try:
        from navsim.agents.para_ssr.ck.temporal import objects as OB
    except Exception as e:                                                    # B1 module not built yet
        pytest.skip(f"objects.py not available ({type(e).__name__})")
    if not hasattr(OB, "ca_forecast_np"):
        pytest.skip("objects.ca_forecast_np missing")
    (obj, mask, fv), _ = TL.rand_objs(3, (14, 9, 4), seed=4, dtype=torch.float64)
    fc = FC.ca_forecast(obj, mask, fv, CA_B.double())
    for i in range(3):
        r = OB.ca_forecast_np(obj[i].numpy(), mask[i].numpy(), fv[i].numpy())
        K = fc.mu.shape[1]
        np.testing.assert_allclose(fc.mu[i, :, 0].numpy(), np.asarray(r["xy"])[:K], rtol=0, atol=1e-5)


def test_oracle_forecast_rules():
    (obj, mask, fv), _ = TL.rand_objs(2, (6, 4), seed=3)
    ca = FC.ca_forecast(obj, mask, fv, CA_B)
    K = ca.mu.shape[1]
    xyh = torch.randn(2, TC.K_OBJ, 10, 3) * 3
    st = torch.zeros(2, TC.K_OBJ, 10, dtype=torch.int8)
    st[:, :, :4] = TC.FC_OBS
    st[:, :, 4:6] = TC.FC_ABSENT
    st[:, :, 6:8] = TC.FC_BEYOND
    xyh[st != TC.FC_OBS] = float("nan")                                       # never read
    fc = FC.oracle_forecast(obj, mask, fv, xyh, st, CA_B)
    ag = ca.agent
    assert torch.isfinite(fc.mu).all()
    assert torch.equal(fc.mu[..., 0, :4, :][ag], xyh[:, :K, :4, :2][ag])
    assert torch.all(fc.b[..., 0, :4, :][ag] == TC.B_ORACLE)
    assert torch.all(fc.present[..., 0, 4:6][ag] == 0) and torch.all(fc.present[..., 0, 6:][ag] == 1)
    assert torch.equal(fc.mu[..., 0, 6:, :], ca.mu[..., 0, 6:, :]) and torch.equal(fc.mu[~ag], ca.mu[~ag])


# ----------------------------------------------------------------------------------------------- learned forecaster
def test_learned_forecaster_shapes_statics_and_modes():
    (obj, mask, fv), _ = TL.rand_objs(2, (12, 7), seed=0)
    o, m, f = FC.trim_objects(obj, mask, fv)
    net = FC.LearnedForecaster(6, seed=0)
    fc = net(o, m, f)
    fc.check()
    T, K, M = fc.mu.shape[:3]
    assert M == 6 and torch.allclose(fc.pi.sum(-1), torch.ones(T, K))
    st = fc.valid & (fc.group == 2)
    assert st.any() and torch.equal(fc.mu[st], fc.c0[st][:, None, None, :].expand(-1, 6, 10, 2))
    assert torch.all(fc.b[st] == torch.tensor(TC.B0)) and torch.all(fc.pi[st][:, 0] == 1)
    ag = fc.agent
    spread = (fc.mu[ag] - fc.mu[ag][:, :1]).abs().amax((1, 2, 3))
    assert (spread > 1e-3).all()                                              # modes differ at init
    assert fc.b.min() >= TC.B_MIN - 1e-7 and fc.b.max() <= TC.B_MAX + 1e-5


def test_learned_forecaster_no_nan_edge_cases():
    net = FC.LearnedForecaster(6, seed=0)
    (obj, mask, fv), _ = TL.rand_objs(2, (3, 1), seed=5)
    cases = {"zero_objects": (obj, torch.zeros_like(mask), fv),
             "one_object": (obj, mask & (torch.arange(TC.K_OBJ) == 0), fv),
             "no_past_frames": (obj, mask, torch.tensor([[True, False, False], [True, False, False]]))}
    for name, (o, m, f) in cases.items():
        o, m, f = FC.trim_objects(o, m, f)
        fc = net(o, m, f)
        fc.check()
        loss, st = forecast_loss(fc, torch.randn(2, TC.K_OBJ, 10, 3), torch.ones(2, TC.K_OBJ, 10, dtype=torch.int8), m[:, 0])
        loss.backward()
        assert math.isfinite(float(loss.detach())), name
        for p in net.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), name
        net.zero_grad()


def test_learned_forecaster_slot_permutation_and_trim_invariance():
    (obj, mask, fv), _ = TL.rand_objs(2, (10, 10), seed=6)
    net = FC.LearnedForecaster(6, seed=0).eval()
    o, m, f = FC.trim_objects(obj, mask, fv)
    with torch.no_grad():
        ref = net(o, m, f)
        full = net(obj, mask, fv)                                             # padded to 64 slots
        K = ref.mu.shape[1]
        torch.testing.assert_close(full.mu[:, :K], ref.mu, rtol=0, atol=1e-5)
        torch.testing.assert_close(full.pi[:, :K], ref.pi, rtol=0, atol=1e-6)
        perm = torch.randperm(K, generator=torch.Generator().manual_seed(0))
        o2 = o.clone()
        m2 = m.clone()
        o2[:, 0] = o[:, 0, perm]
        m2[:, 0] = m[:, 0, perm]
        out = net(o2, m2, f)
        torch.testing.assert_close(out.mu, ref.mu[:, perm], rtol=0, atol=1e-5)
        again = net(o, m, f)
    assert torch.equal(again.mu, ref.mu) and torch.equal(again.logit_pi, ref.logit_pi)   # deterministic


# ----------------------------------------------------------------------------------------------- raster known answers
def _geom(*cands):
    flat = torch.stack(list(cands))
    return corridor_geometry(flat), flat


def test_raster_static_object_on_path_known_answer():
    geom, flat = _geom(TL.straight([5.0 * k for k in range(1, 9)]))           # 10 m/s, S_8 = 40, S_look = 50
    ds = 50.0 / 48
    assert abs(float(geom.S_look[0]) - 50.0) < 1e-4
    x_obj = 17.3
    fc = fc_static((x_obj, 0.0), L=0.5, W=0.5, group=2)
    R = RA.rasterize(fc, geom, 1, 1, TC.ROI["bevfusion"])
    ch = {n: i for i, n in enumerate(RA.RASTER_CHANNELS)}
    H = TC.EGO_W / 2 + 0.25
    p_lat = 1 - math.exp(-H / TC.B0)
    u = x_obj / ds - 0.5
    j = int(math.floor(u))
    f = u - j
    occ = R[0, ch["st_occ"]]
    for k in range(21):
        assert abs(float(occ[j, k]) - (1 - f) * p_lat) < 1e-5 and abs(float(occ[j + 1, k]) - f * p_lat) < 1e-5
    assert abs(float(occ.sum()) - 21 * p_lat) < 1e-4
    np.testing.assert_allclose(R[0, ch["st_inc"]].numpy(), occ.numpy() * min(H, 3) / 3, atol=1e-6)
    np.testing.assert_allclose(R[0, ch["st_ext"]].numpy(), occ.numpy() * 0.25 / 5, atol=1e-6)
    for n in RA.RASTER_CHANNELS[5:15]:
        assert float(R[0, ch[n]].abs().sum()) == 0.0, n


def test_raster_lateral_overlap_and_moving_vehicle():
    geom, flat = _geom(TL.straight([5.0 * k for k in range(1, 9)]))
    ch = {n: i for i, n in enumerate(RA.RASTER_CHANNELS)}
    # lateral: static 1x1 box 2.5 m left of the path -> Laplace overlap of [-H, H], no incursion
    fc = fc_static((20.0, 2.5), L=1.0, W=1.0, group=2)
    R = RA.rasterize(fc, geom, 1, 1, TC.ROI["bevfusion"])
    H, d, b = TC.EGO_W / 2 + 0.5, 2.5, TC.B0
    F = lambda z: 0.5 * math.exp(z / b) if z < 0 else 1 - 0.5 * math.exp(-z / b)   # noqa: E731
    assert abs(float(R[0, ch["st_occ"], :, 3].sum()) - (F(H - d) - F(-H - d))) < 1e-7
    assert float(R[0, ch["st_inc"]].abs().sum()) == 0.0
    # vehicle ahead at 10 m moving +x at 5 m/s (ego 10 m/s): b = 0.5 -> P_lat = 1 - exp(-H / b) after the first knot
    fc = fc_static((10.0, 0.0), L=4.5, W=1.9, group=0, vel=(5.0, 0.0), b=0.5)
    R = RA.rasterize(fc, geom, 1, 1, TC.ROI["bevfusion"])
    H = TC.EGO_W / 2 + 0.95
    for k in (4, 12, 16):                                                     # t = 1, 3, 4 s (t = 2: rear axle level)
        t = 0.25 * k
        occ = float(R[0, ch["veh_occ"], :, k].sum())
        assert abs(occ - (1 - math.exp(-H / 0.5))) < 1e-5, k
        s_obj, s_ego = 10.0 + 5 * t, 10.0 * t
        vc = float(R[0, ch["veh_vclose"], :, k].sum()) / occ
        assert abs(vc - math.copysign(1, s_obj - s_ego) * (10.0 - 5.0) / 15) < 1e-5, k
        assert abs(float(R[0, ch["veh_dapp"], :, k].sum())) < 1e-6
    assert float(R[0, ch["vru_occ"]].abs().sum()) == 0.0


def test_raster_ego_channels_and_known_mask():
    geom, flat = _geom(TL.straight([5.0 * k for k in range(1, 9)]))
    ds = 50.0 / 48
    R = RA.rasterize(fc_static((100.0, 0.0)), geom, 1, 1, TC.ROI["bevfusion"])
    lo_b, hi_b = np.arange(48) * ds, (np.arange(48) + 1) * ds
    cov = lambda lo, hi: np.clip(np.minimum(hi, hi_b) - np.maximum(lo, lo_b), 0, ds) / ds   # noqa: E731
    for k in range(21):
        t = 0.25 * k
        se = 10.0 * t
        np.testing.assert_allclose(R[0, 1, :, k].numpy(), cov(se + TC.EGO_REAR, se + TC.EGO_FRONT), atol=1e-5)
        for c, dl in zip((2, 3, 4), (0.3, 0.6, 0.9)):
            exp = cov(se + 10 * dl + TC.EGO_REAR, se + 10 * dl + TC.EGO_FRONT) if t <= 4.0 else np.zeros(48)
            np.testing.assert_allclose(R[0, c, :, k].numpy(), exp, atol=1e-5)
    known = R[0, 0, :, 0].numpy()
    exp = (hi_b <= 32.0).astype(np.float32)                                   # x of the station's far edge <= 32 m
    np.testing.assert_array_equal(known, exp)
    assert known[:30].all() and not known[30:].any() and torch.all(R[0, 0] == R[0, 0, :, :1])
    assert float(R[0, 5:].abs().sum()) == 0.0                                 # object beyond the corridor


def test_raster_variants_with_identical_stations_share_agent_channels():
    # same path and S_look (S_8 = 40, last segment 5 m), different speed profile -> identical stations
    a = TL.straight([5.0 * k for k in range(1, 9)])
    b = TL.straight([3.0, 7.0, 12.0, 17.0, 22.0, 28.0, 35.0, 40.0])
    geom, flat = _geom(a, b)
    (obj, mask, fv), _ = TL.rand_objs(1, (15,), seed=7)
    fc = FC.ca_forecast(*FC.trim_objects(obj, mask, fv), CA_B)
    R = RA.rasterize(fc, geom, 1, 2, TC.ROI["bevfusion"])
    assert float(R[:, 5:].abs().sum()) > 0
    ch = {n: i for i, n in enumerate(RA.RASTER_CHANNELS)}
    shared = [ch[n] for n in RA.RASTER_CHANNELS[5:] if not n.endswith("vclose")]
    torch.testing.assert_close(R[0, shared], R[1, shared], rtol=0, atol=1e-5)
    vc = [ch["veh_vclose"], ch["vru_vclose"]]                                 # closing speed uses the ego schedule
    assert float((R[0, vc] - R[1, vc]).abs().max()) > 1e-3
    torch.testing.assert_close(R[0, 0], R[1, 0], rtol=0, atol=0)
    assert float((R[0, 1:5] - R[1, 1:5]).abs().max()) > 0.1                  # ego schedules differ
    # each candidate rasterised alone gives the same channels (no cross-candidate leakage)
    R0 = RA.rasterize(fc, corridor_geometry(flat[:1]), 1, 1, TC.ROI["bevfusion"])
    torch.testing.assert_close(R0[0], R[0], rtol=0, atol=0)


def test_raster_zero_objects_and_gradient_to_mu():
    geom, flat = _geom(*TL.rand_cands(1, 3, seed=1))
    (obj, mask, fv), _ = TL.rand_objs(1, (8,), seed=2)
    o, m, f = FC.trim_objects(obj, torch.zeros_like(mask), fv)
    fc = FC.ca_forecast(o, m, f, CA_B)
    R = RA.rasterize(fc, geom, 1, 3, TC.ROI["bevfusion"])
    G = RA.gap_scalars(fc, flat, 1, 3, raw=True)
    assert torch.isfinite(R).all() and float(R[:, 5:].abs().sum()) == 0 and float(R[:, 1].sum()) > 0
    exp = torch.tensor([15.0] * 6 + [4.5, 4.5, 0.0])
    assert torch.equal(G[:, :9], exp.expand(3, 9)) and torch.equal(G[:, 9:18], exp.expand(3, 9))
    assert torch.equal(G[:, 18:], torch.full((3, 2), 15.0))
    # gradient: d raster / d mu, b, pi is non-zero (raster is differentiable in the forecast)
    fc = FC.ca_forecast(*FC.trim_objects(obj, mask, fv), CA_B)
    fc.mu.requires_grad_(True)
    fc.b.requires_grad_(True)
    geom, flat = _geom(TL.straight([5.0 * k for k in range(1, 9)]), *TL.rand_cands(1, 2, seed=3))
    R = RA.rasterize(fc, geom, 1, 3, TC.ROI["bevfusion"])
    (R[:, 5:15] * torch.linspace(0, 1, 48)[:, None]).sum().backward()
    assert fc.mu.grad is not None and float(fc.mu.grad.abs().sum()) > 0
    assert float(fc.b.grad.abs().sum()) > 0 and torch.isfinite(fc.mu.grad).all()


# ----------------------------------------------------------------------------------------------- gap scalars vs a3
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gap_scalars_equal_a3_reduction(A3, seed):
    g = np.random.default_rng(seed)
    frames = TL.rand_token(g, int(g.integers(4, 20)))
    obj, mask, fv = TL.pack_tokens([frames])
    fc = FC.ca_forecast(*FC.trim_objects(obj, mask, fv), CA_B)
    cands = TL.rand_cands(1, 12, seed=seed)
    cands[0] = TL.straight([5.0 * k for k in range(1, 9)])
    G = RA.gap_scalars(fc, cands, 1, 12, raw=True).double().numpy()
    b0, l0 = TL.rows_to_ft(obj[0, 0, :int(mask[0, 0].sum())].double().numpy())
    n = len(b0)
    mu = fc.mu[0, :n, 0].double().numpy()
    world_ca = lambda tau: TL.np_knot_world(mu, b0[:, :2], tau)              # noqa: E731
    ref, _ = A3.Worker.__call__(TL.FakeCall(A3, b0, l0, world_ca), ("t0", cands.double().numpy(), 0.0))
    np.testing.assert_allclose(G[:, :9], ref[:, 0], rtol=0, atol=1e-4)        # S world
    np.testing.assert_allclose(G[:, 9:18], ref[:, 2], rtol=0, atol=1e-4)      # forecast world (M = 1)
    np.testing.assert_allclose(G[:, 18], G[:, 9], atol=0)
    np.testing.assert_allclose(G[:, 19], G[:, 12], atol=0)
    assert (ref[:, 0, 0] < 15).any()                                          # non-trivial scene
    # normalised output
    Gn = RA.gap_scalars(fc, cands, 1, 12).double().numpy()
    np.testing.assert_allclose(Gn[:, :6], G[:, :6] / 5, atol=1e-6)
    np.testing.assert_allclose(Gn[:, 6:8], G[:, 6:8] / 4.5, atol=1e-6)
    np.testing.assert_allclose(Gn[:, 8], G[:, 8] / 16, atol=1e-6)


def test_gap_scalars_multimode_equal_numpy_reference(A3):
    (obj, mask, fv), _ = TL.rand_objs(1, (14,), seed=11)
    o, m, f = FC.trim_objects(obj, mask, fv)
    torch.manual_seed(0)
    net = FC.LearnedForecaster(3, seed=0)
    with torch.no_grad():
        for p in net.dec_out.parameters():
            p.mul_(30.0)                                                      # spread the modes (metres)
        for p in net.dec_pi.parameters():
            p.mul_(20.0)
        fc = net(o, m, f)
    cands = TL.rand_cands(1, 10, seed=5)
    G = RA.gap_scalars(fc, cands, 1, 10, raw=True).double().numpy()
    v = fc.valid[0]
    tau = (np.asarray(TC.TJ)[:, None] + np.asarray(TC.TAPS)[None]).ravel()
    fa = RA.forecast_at(fc, torch.as_tensor(tau, dtype=torch.float32))
    ref = TL.np_gap_general(A3, cands.double().numpy(), fc.c0[0][v].double().numpy(), fc.h0[0][v].double().numpy(),
                            fc.L[0][v].double().numpy(), fc.W[0][v].double().numpy(), fc.agent[0][v].numpy(),
                            (v & (fc.group[0] == 2))[v].numpy(), fa["xy"][0][v].double().numpy(),
                            fc.pi[0][v].double().numpy(), fa["present"][0][v].numpy() > 0)
    np.testing.assert_allclose(G, ref, rtol=0, atol=1e-4)
    assert (np.abs(G[:, 9:18] - G[:, 18:19]) > 1e-3).any() or (G[:, 18] <= G[:, 9] + 1e-6).all()
    assert (G[:, 18] <= G[:, 9] + 1e-4).all() and (G[:, 19] <= G[:, 12] + 1e-4).all()   # worst <= expected


def test_forecast_at_matches_numpy_interp():
    (obj, mask, fv), _ = TL.rand_objs(1, (6,), seed=2)
    net = FC.LearnedForecaster(2, seed=0)
    with torch.no_grad():
        fc = net(*FC.trim_objects(obj, mask, fv))
    t = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.55, 1.75, 3.9, 4.9, 5.0])
    fa = RA.forecast_at(fc, t)
    K = fc.mu.shape[1]
    for k in range(K):
        for mm in range(2):
            ref = TL.np_knot_world(fc.mu[0, k, mm][None].double().numpy(), fc.c0[0, k][None].double().numpy(),
                                   t.double().numpy())[:, 0]
            np.testing.assert_allclose(fa["xy"][0, k, mm].double().numpy(), ref, atol=1e-5)
    assert torch.all(fa["present"] == 1)


# ----------------------------------------------------------------------------------------------- loss
def test_forecast_loss_known_answer_and_zero_targets():
    y = torch.zeros(1, TC.K_OBJ, 10, 3)
    y[0, 0, :, 0] = torch.arange(10.0)
    mu = torch.zeros(1, 1, 2, 10, 2)
    mu[0, 0, 0, :, 0] = y[0, 0, :, 0] + 0.1
    mu[0, 0, 1, :, 0] = y[0, 0, :, 0] + 1.0
    mu[0, 0, 1, :, 1] = 1.0
    mu.requires_grad_(True)
    log_b = torch.zeros(1, 1, 2, 10, 2, requires_grad=True)
    logit_pi = torch.zeros(1, 1, 2, requires_grad=True)
    fc = FC.Forecast(mu=mu, b=log_b.exp(), pi=torch.softmax(logit_pi, -1), present=torch.ones(1, 1, 2, 10),
                     c0=torch.zeros(1, 1, 2), v0=torch.zeros(1, 1, 2), h0=torch.zeros(1, 1), L=torch.ones(1, 1),
                     W=torch.ones(1, 1), group=torch.zeros(1, 1, dtype=torch.long),
                     valid=torch.ones(1, 1, dtype=torch.bool), log_b=log_b, logit_pi=logit_pi)
    st = torch.zeros(1, TC.K_OBJ, 10, dtype=torch.int8)
    st[0, 0, :2] = TC.FC_OBS
    st[0, 0, 2:4] = TC.FC_ABSENT
    y[0, 0, 2:] = float("nan")                                                # unsupervised steps: never read
    loss, s = forecast_loss(fc, y, st, torch.ones(1, TC.K_OBJ, dtype=torch.bool))
    exp = math.log(2) + 0.05 + math.log(2)                                    # NLL (b = 1) + CE (pi = 1/2)
    assert abs(float(loss) - exp) < 1e-6 and s["fc_n_obj"] == 1
    assert abs(s["fc_minade"] - 0.1) < 1e-6 and abs(s["fc_minfde"] - 0.1) < 1e-6
    loss.backward()
    assert float(mu.grad[0, 0, 1].abs().sum()) == 0 and float(mu.grad[0, 0, 0, :2].abs().sum()) > 0
    assert torch.isfinite(mu.grad).all() and float(logit_pi.grad.abs().sum()) > 0
    # dropped input object (obj_mask0 False) and zero targets -> finite 0 with gradients present
    for mask0, st0 in ((torch.zeros(1, TC.K_OBJ, dtype=torch.bool), st), (torch.ones(1, TC.K_OBJ, dtype=torch.bool),
                                                                     torch.zeros_like(st))):
        for p in (mu, log_b, logit_pi):
            p.grad = None
        loss, s = forecast_loss(fc, y, st0, mask0)
        assert float(loss) == 0.0 and s["fc_n_obj"] == 0
        loss.backward()
        for p in (mu, log_b, logit_pi):
            assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) == 0


# ----------------------------------------------------------------------------------------------- branch
def test_branch_init_param_counts_and_shared_identity():
    br = {a: branch(a) for a in "abd"}
    assert br["a"].param_counts() == {"temporal_shared": 237888, "forecaster": br["a"].param_counts()["forecaster"]}
    assert br["b"].param_counts() == {"temporal_shared": 237888, "forecaster": 0}
    assert br["d"].param_counts() == {"temporal_shared": 237888, "forecaster": 0}
    sa = br["a"].state_dict()
    for a in "bd":
        sb = br[a].state_dict()
        assert set(sb) == {k for k in sa if not k.startswith("forecaster.")} | {"ca_b"}
        for k, v in sb.items():
            if k != "ca_b":
                assert torch.equal(v, sa[k]), k
        assert torch.equal(sb["ca_b"], CA_B)
    for m in (br["a"].st_proj, br["a"].gap_proj):
        assert float(m.weight.abs().sum()) == 0 and float(m.bias.abs().sum()) == 0
    torch.manual_seed(99)
    x = torch.rand(3)
    torch.manual_seed(99)
    branch("a", seed=0)
    assert torch.equal(torch.rand(3), x)                                      # no global RNG consumed
    b2 = branch("a", seed=0)
    for (k, v), (k2, v2) in zip(sa.items(), b2.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)                                 # deterministic
    b3 = branch("a", seed=1)
    assert not torch.equal(b3.st_enc[0].weight, br["a"].st_enc[0].weight)


def _features(br, objs, cands, T, Kc):
    ctx = br.scene(objs)
    geom = corridor_geometry(cands)
    return ctx, br.features(ctx, geom, cands, T, Kc)


def _randomize(br, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in (br.st_proj, br.gap_proj):
            m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.05)
            m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.05)


@pytest.mark.parametrize("arm", ["a", "b"])
def test_branch_zero_init_noop_and_gradient_paths(arm):
    T, Kc = 2, 6
    (obj, mask, fv), _ = TL.rand_objs(T, (12, 7), seed=0)
    cands = TL.rand_cands(T, Kc, seed=0)
    br = branch(arm)
    ctx, (st, ud) = _features(br, objs_dict(obj, mask, fv), cands, T, Kc)
    assert st.shape == (T * Kc, 48, 192) and ud.shape == (T * Kc, 192)
    assert float(st.abs().max()) == 0.0 and float(ud.abs().max()) == 0.0       # untrained branch = no-op
    # every parameter finite (possibly zero) gradient: critic proxy + forecast loss
    loss = (st * torch.randn_like(st)).sum() + (ud * torch.randn_like(ud)).sum()
    if arm == "a":
        fl, _ = forecast_loss(ctx.fc_raw, torch.randn(T, TC.K_OBJ, 10, 3) * 5, torch.ones(T, TC.K_OBJ, 10, dtype=torch.int8),
                              mask[:, 0])
        loss = loss + fl
    loss.backward()
    for n, p in br.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert float(br.st_proj.weight.grad.abs().sum()) > 0 and float(br.gap_proj.weight.grad.abs().sum()) > 0
    if arm == "a":
        assert float(br.forecaster.dec_out.weight.grad.abs().sum()) > 0


def test_branch_fc_backprop_switch():
    T, Kc = 2, 5
    (obj, mask, fv), _ = TL.rand_objs(T, (12, 7), seed=1)
    cands = TL.rand_cands(T, Kc, seed=1)
    for backprop in (False, True):
        br = branch("a", fc_backprop=backprop)
        _randomize(br)
        ctx, (st, ud) = _features(br, objs_dict(obj, mask, fv), cands, T, Kc)
        (st.sum() + ud.sum()).backward()
        g = [p.grad for p in br.forecaster.parameters()]
        if backprop:
            assert any(x is not None and float(x.abs().sum()) > 0 for x in g)  # raster -> forecaster gradient
            assert all(x is None or torch.isfinite(x).all() for x in g)
        else:
            assert all(x is None for x in g)                                  # critic never reaches the forecaster
        assert float(br.st_enc[0].weight.grad.abs().sum()) > 0


def test_branch_causality_and_input_guards():
    T, Kc = 2, 4
    (obj, mask, fv), _ = TL.rand_objs(T, (10, 6), seed=2)
    cands = TL.rand_cands(T, Kc, seed=2)
    cfg = cfg_of("a")
    br = branch("a").eval()
    _randomize(br)
    batch = {"obj_in": obj, "obj_mask": mask, "frame_valid": fv, "fc_xyh": torch.randn(T, TC.K_OBJ, 10, 3),
             "fc_state": torch.ones(T, TC.K_OBJ, 10, dtype=torch.int8), "priv_fc_xyh": torch.randn(T, TC.K_OBJ, 10, 3),
             "priv_fc_state": torch.ones(T, TC.K_OBJ, 10, dtype=torch.int8), "lead_decel": torch.ones(T)}
    with torch.no_grad():
        _, ref = _features(br, objs_from_batch(batch, cfg), cands, T, Kc)
        pert = dict(batch, fc_xyh=batch["fc_xyh"] + 7, fc_state=torch.zeros_like(batch["fc_state"]),
                    priv_fc_xyh=-batch["priv_fc_xyh"], lead_decel=torch.zeros(T))
        _, out = _features(br, objs_from_batch(pert, cfg), cands, T, Kc)
    assert torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1])      # future-only data cannot change outputs
    assert set(objs_from_batch(batch, cfg)) == set(TC.OBJ_INPUT_KEYS)
    assert set(objs_from_batch(batch, cfg_of("d"))) == set(TC.OBJ_INPUT_KEYS + TC.PRIV_INPUT_KEYS)
    assert objs_from_batch(batch, TC.TemporalCfg()) is None
    o = objs_dict(obj, mask, fv)
    with pytest.raises(ValueError):
        br.scene(dict(o, fc_xyh=batch["fc_xyh"]))                             # targets never reach forward
    with pytest.raises(ValueError):
        br.scene(dict(o, priv_fc_xyh=batch["priv_fc_xyh"], priv_fc_state=batch["priv_fc_state"]))
    with pytest.raises(ValueError):
        branch("d").scene(o)                                                  # oracle needs the privileged keys
    with pytest.raises(ValueError):
        br.scene({"obj_in": obj, "obj_mask": mask})
    bad = obj.clone()
    bad[0, 1, 0, TC.COL["age"]] = -0.5                                        # a FUTURE frame in a past slot
    with pytest.raises(ValueError, match="non-causal"):
        br.scene(objs_dict(bad, mask, fv))
    bad = obj.clone()
    bad[1, 0, 0, TC.COL["age"]] = 0.5                                         # t0 slot carrying an old frame
    with pytest.raises(ValueError, match="non-causal"):
        br.scene(objs_dict(bad, mask, fv))
    # oracle arm IS privileged: its features follow the GT futures (sanity of the oracle path)
    bd = branch("d").eval()
    _randomize(bd)
    with torch.no_grad():
        _, r1 = _features(bd, objs_from_batch(batch, cfg_of("d")), cands, T, Kc)
        _, r2 = _features(bd, objs_from_batch(pert, cfg_of("d")), cands, T, Kc)
    assert not torch.equal(r1[0], r2[0])


@pytest.mark.parametrize("arm", ["a", "b"])
def test_branch_candidate_permutation_trim_and_determinism(arm):
    T, Kc = 2, 6
    (obj, mask, fv), _ = TL.rand_objs(T, (11, 5), seed=3)
    cands = TL.rand_cands(T, Kc, seed=3)
    br = branch(arm).eval()
    _randomize(br)
    o = objs_dict(obj, mask, fv)
    with torch.no_grad():
        ctx, (st, ud) = _features(br, o, cands, T, Kc)
        perm = torch.cat([torch.randperm(Kc, generator=torch.Generator().manual_seed(t)) + t * Kc for t in range(T)])
        _, (st2, ud2) = _features(br, o, cands[perm], T, Kc)
        torch.testing.assert_close(st2, st[perm], rtol=0, atol=1e-6)
        torch.testing.assert_close(ud2, ud[perm], rtol=0, atol=1e-6)
        K = int(torch.nonzero(mask.reshape(-1, TC.K_OBJ).any(0)).max()) + 1
        _, (st3, ud3) = _features(br, objs_dict(obj[:, :, :K], mask[:, :, :K], fv), cands, T, Kc)
        _, (st4, ud4) = _features(br, o, cands, T, Kc)
    tol = 0 if arm == "b" else 1e-5
    torch.testing.assert_close(st3, st, rtol=0, atol=tol)
    torch.testing.assert_close(ud3, ud, rtol=0, atol=tol)
    assert torch.equal(st4, st) and torch.equal(ud4, ud)                      # deterministic
    assert ctx.n_obj == int(torch.nonzero(mask[:, 0].any(0)).max()) + 1
    assert trim_forecast(ctx.fc) is ctx.fc


def test_branch_forecast_shared_by_candidate_sets():
    """scene() runs once per token set; cand / extra candidate sets of a token read the same forecast."""
    T = 2
    (obj, mask, fv), _ = TL.rand_objs(T, (9, 9), seed=4)
    br = branch("a").eval()
    _randomize(br)
    c1, c2 = TL.rand_cands(T, 3, seed=4), TL.rand_cands(T, 5, seed=5)
    with torch.no_grad():
        ctx = br.scene(objs_dict(obj, mask, fv))
        f1 = br.features(ctx, corridor_geometry(c1), c1, T, 3)
        f2 = br.features(ctx, corridor_geometry(c2), c2, T, 5)
        both = torch.cat([c1.reshape(T, 3, 8, 3), c2.reshape(T, 5, 8, 3)], 1).reshape(T * 8, 8, 3)
        f3 = br.features(ctx, corridor_geometry(both), both, T, 8)
    i1 = torch.cat([torch.arange(3) + 8 * t for t in range(T)])
    i2 = torch.cat([torch.arange(3, 8) + 8 * t for t in range(T)])
    torch.testing.assert_close(f3[0][i1], f1[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(f3[0][i2], f2[0], rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        br.hook(ctx, T, 3)(corridor_geometry(c2), c2)


# ----------------------------------------------------------------------------------------------- trunk integration
def _ck_inputs(T, K, seed=0):
    from navsim.agents.para_ssr.refiner import e2e as E
    g = torch.Generator().manual_seed(seed)
    cand = TL.rand_cands(T, K, seed=seed).reshape(T, K, 8, 3)
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = 8.0
    bev = torch.randn(T, 256, 50, 100, generator=g)
    return bev, cand, sf, E.ego_inputs(sf)


def _norm():
    rng = np.random.default_rng(3)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32)


@pytest.mark.parametrize("arm", ["a", "b"])
def test_trunk_hook_on_at_init_equals_off_and_gradients(arm):
    from navsim.agents.para_ssr.ck import model as M
    T, K = 2, 4
    net = M.CKNet("T", seed=0, norm=_norm()).eval()
    br = branch(arm, seed=0).eval()
    bev, cand, sf, (v0, a0, eds, cmd) = _ck_inputs(T, K)
    (obj, mask, fv), _ = TL.rand_objs(T, (10, 4), seed=0)
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, T)
        off = net.trunk.candidates(feat, mem, cand, v0, a0, eds, cmd)
        ctx = br.scene(objs_dict(obj, mask, fv))
        on = TL.candidates_hooked(net.trunk, feat, mem, cand, v0, a0, eds, cmd, br.hook(ctx, T, K))
    for k in ("h", "z_lon", "w_lat", "gate_logit", "ud"):
        assert torch.equal(on[k], off[k]), k                                  # zero-init branch: bit-identical
    net.trunk._temporal_required = True
    with pytest.raises(RuntimeError):
        TL.candidates_hooked(net.trunk, feat, mem, cand, v0, a0, eds, cmd, None)
    # gradients of a score-head loss reach trunk, branch encoder and (randomised) projections
    net.train()
    br.train()
    _randomize(br)
    feat, mem = net.trunk.scene(bev, T)
    ctx = br.scene(objs_dict(obj, mask, fv))
    o = TL.candidates_hooked(net.trunk, feat, mem, cand, v0, a0, eds, cmd, br.hook(ctx, T, K))
    net.score(o, T, K).sum().backward()
    assert float(net.trunk.station.weight.grad.abs().sum()) > 0
    for p in (br.st_enc[0].weight, br.st_proj.weight, br.gap_proj.weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0 and torch.isfinite(p.grad).all()
    if arm == "a":
        assert all(p.grad is None for p in br.forecaster.parameters())        # fc_backprop False


# ----------------------------------------------------------------------------------------------- speed (CPU)
def test_cpu_speed_record():
    """Records CPU timings (printed with -s): scene per token, features per candidate.  Loose sanity bounds only."""
    torch.set_num_threads(1)
    T, Kc = 2, 32
    (obj, mask, fv), _ = TL.rand_objs(T, (40, 30), seed=9, n_distract=10)
    cands = TL.rand_cands(T, Kc, seed=9)
    geom = corridor_geometry(cands)
    res = {}
    for arm in ("a", "b"):
        br = branch(arm).eval()
        with torch.no_grad():
            br.scene(objs_dict(obj, mask, fv))
            t0 = time.perf_counter()
            ctx = br.scene(objs_dict(obj, mask, fv))
            t1 = time.perf_counter()
            br.features(ctx, geom, cands, T, Kc)
            t2 = time.perf_counter()
        res[arm] = dict(n_obj=ctx.n_obj, scene_ms_per_token=1e3 * (t1 - t0) / T,
                        features_ms_per_cand=1e3 * (t2 - t1) / (T * Kc))
    print("\n[v3 temporal CPU speed, 1 thread]", res)
    assert res["b"]["features_ms_per_cand"] < 200 and res["a"]["scene_ms_per_token"] < 2000


def test_load_ca_b_from_json_and_branch_build():
    """ca_scales.json (fit_ca_scales.py format) -> persistent buffer ca_b of arms b / d; bad files refused."""
    import json
    import shutil
    import uuid
    from navsim.agents.para_ssr.ck.temporal.branch import load_ca_b
    d = TC.D3 / "test_tmp" / f"cab_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        good = {"version": "x", "split": "navtrain_train", "b": {"vehicle": [0.1 * (k + 1) for k in range(10)],
                                                                "vru": [0.05 * (k + 1) for k in range(10)]}}
        (d / "ok.json").write_text(json.dumps(good))
        cb = load_ca_b(d / "ok.json")
        assert cb.shape == (2, 10) and abs(float(cb[0, 9]) - 1.0) < 1e-6
        br = TemporalBranch(TC.TemporalCfg(on=True, forecaster="ca", n_modes=1, ca_scales=str(d / "ok.json")))
        assert torch.equal(br.ca_b, cb) and "ca_b" in br.state_dict()
        bad = dict(good, b={"vehicle": [0.1] * 9, "vru": [0.1] * 10})
        (d / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_ca_b(d / "bad.json")
        with pytest.raises(FileNotFoundError):
            TemporalBranch(TC.TemporalCfg(on=True, forecaster="ca", n_modes=1, ca_scales=str(d / "missing.json")))
    finally:
        shutil.rmtree(d, ignore_errors=True)
