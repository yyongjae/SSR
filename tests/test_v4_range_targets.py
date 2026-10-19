"""refiner v4_3 M2 range targets (navsim/agents/para_ssr/ck/v4/range_targets.py): blocked field (exact SAT vs a dense
brute force, numpy twin, time pooling, not-behind, index / mask / known, LON field), margins (analytic roads, grid vs
SDF, unread edge), v_max (table inversion + provenance refit when the F1 data exist), windows (incl. an all-unknown
parent), executed slack (NC / TTC / DAC, stopped-ego skips with |v|, conservative read, TTC not-behind from the current
ego, exact frames at 30 deg and on a ring, float32 = float64 bins, exact free slack, DAC critical state off the grid /
unreadable, probability input).  CPU, synthetic, ~10 s.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_range_targets.py
"""
import math
import os

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck.v4 import range_targets as RT
from navsim.agents.para_ssr.refiner import gt_future as GF
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry, lateral_offsets
from navsim.agents.para_ssr.refiner.geometry import HALF_LEN, HALF_WID, RA2C
from navsim.agents.para_ssr.refiner.sdf import ego_corners, grid_mesh, sample_sdf
from navsim.agents.para_ssr.refiner.surrogate import box_gaps

F64 = torch.float64
FEATS = "/home/external-user/ssd/yongjae_refiner/ck/ck2/analysis/detmap_path_ideas/F1/out/val/feats.parquet"


def _draft(v, kap, n=8):
    t = torch.arange(1, n + 1, dtype=F64) * 0.5
    s = v * t
    if abs(kap) < 1e-9:
        return torch.stack([s, 0 * s, 0 * s], -1)
    return torch.stack([torch.sin(kap * s) / kap, (1 - torch.cos(kap * s)) / kap, kap * s], -1)


def _static(x, y, h=0.0, L=4.0, W=2.0, T=50):
    """one box held at (x, y, h) for T samples -> [1, T, 5]."""
    return torch.tensor([x, y, h, L, W], dtype=F64).expand(1, T, 5).clone()


def _brute(fr, boxes, obs, t, tbin, M, behind_m=RT.BEHIND_M, lon_max=False):
    """dense reference: every cell x object x sample (no culling); lon_max: also the LON field (max lon of the
    overlapping OBS objects, -inf none)."""
    P = fr["pts"].shape[0]
    out = torch.zeros(P, 48, 17, M, dtype=torch.bool)
    lout = torch.full((P, 48, 17, M), -math.inf, dtype=fr["pts"].dtype)
    mt = torch.as_tensor(RT.time_bins(t, tbin))
    eh = torch.stack([torch.cos(fr["h"]), torch.sin(fr["h"])], -1)
    for p in range(P):
        ego = torch.cat([fr["pts"][p], fr["h"][p][:, None, None].expand(48, 17, 1)], -1)        # [48, 17, 3]
        g = box_gaps(ego[:, :, None, None, :], boxes[None, None]).max(-1).values <= 0         # [48, 17, A, T]
        ce = fr["pts"][p] + RA2C * eh[p][:, None, :]
        lon = ((boxes[None, None, :, :, :2] - ce[:, :, None, None, :]) * eh[p][:, None, None, None, :]).sum(-1)
        hit = g & obs[None, None] & (lon >= behind_m if behind_m is not None else True)
        lv = torch.where(g & obs[None, None], lon, torch.full_like(lon, -math.inf))
        for m in range(M):
            sel = mt == m
            if sel.any():
                out[p, :, :, m] = hit[..., sel].flatten(2).any(-1)
                lout[p, :, :, m] = lv[..., sel].flatten(2).max(-1).values
    return (out, lout) if lon_max else out


def _line_frame(theta, x0=0.0, y0=0.0, ds=1.0, dtype=F64):
    """exact parent frame (parent_frame dict) of a straight path from (x0, y0) at heading theta, stations
    (j + 0.5) ds."""
    s = (torch.arange(48, dtype=dtype) + 0.5) * ds
    e = torch.tensor([math.cos(theta), math.sin(theta)], dtype=dtype)
    nrm = torch.tensor([-math.sin(theta), math.cos(theta)], dtype=dtype)
    p = torch.tensor([x0, y0], dtype=dtype) + s[:, None] * e
    d = lateral_offsets(dtype=dtype)
    return {"pts": (p[:, None, :] + d[None, :, None] * nrm)[None], "p": p[None], "u": e.expand(48, 2)[None].clone(),
            "n": nrm.expand(48, 2)[None].clone(), "h": torch.full((1, 48), theta, dtype=dtype), "s": s[None],
            "ds": torch.tensor([ds], dtype=dtype), "kappa": torch.zeros(1, 48, dtype=dtype)}


def _ring_frame(R, ds=1.0, dtype=F64):
    """exact parent frame of a left-turning circle of radius R through the origin (heading 0, centre (0, R))."""
    s = (torch.arange(48, dtype=dtype) + 0.5) * ds
    ph = s / R
    p = torch.stack([R * torch.sin(ph), R * (1 - torch.cos(ph))], -1)
    nrm = torch.stack([-torch.sin(ph), torch.cos(ph)], -1)
    d = lateral_offsets(dtype=dtype)
    return {"pts": (p[:, None, :] + d[None, :, None] * nrm[:, None, :])[None], "p": p[None],
            "u": torch.stack([torch.cos(ph), torch.sin(ph)], -1)[None], "n": nrm[None], "h": ph[None], "s": s[None],
            "ds": torch.tensor([ds], dtype=dtype), "kappa": torch.full((1, 48), 1.0 / R, dtype=dtype)}


def _box(x, y, h, L=4.0, W=2.0, T=50):
    return torch.tensor([x, y, h, L, W], dtype=F64).expand(1, T, 5).clone()


def _random_scene(seed, P=3, A=12, T=50):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(P, 2, generator=g, dtype=F64)
    par = torch.stack([_draft(2 + 9 * float(u[k, 0]), 0.08 * (float(u[k, 1]) - 0.5)) for k in range(P)])
    tt = torch.arange(T, dtype=F64) * 0.1
    c = torch.rand(A, 1, 2, generator=g, dtype=F64) * torch.tensor([45.0, 16.0], dtype=F64) \
        - torch.tensor([3.0, 8.0], dtype=F64)
    vel = (torch.rand(A, 1, 2, generator=g, dtype=F64) - 0.5) * 12
    xy = c + vel * tt[None, :, None]
    h = torch.rand(A, 1, generator=g, dtype=F64).expand(A, T) * 6.3
    LW = torch.rand(A, 1, 2, generator=g, dtype=F64).expand(A, T, 2) * torch.tensor([4.0, 1.5], dtype=F64) + 0.5
    boxes = torch.cat([xy, h[..., None], LW], -1)
    obs = torch.rand(A, T, generator=g) > 0.2
    return par, boxes, obs, tt


# ----------------------------------------------------------------------------------------------- blocked field
def test_static_box_exact_cells():
    g = corridor_geometry(_draft(5.0, 0.0)[None])                 # straight, ds = 1 m, s_j = j + 0.5
    B = RT.blocked_field(g, _static(20.0, 0.0)[None], torch.ones(1, 1, 50, dtype=torch.bool), details=True)
    blk = B["blocked"]
    assert blk.shape == (1, 48, 17, 20) and B["known"].all()
    d = lateral_offsets(dtype=F64)
    s = g.s[0]
    # overlap iff |s_j + RA2C - 20| <= HALF_LEN + 2 and |d_i| <= HALF_WID + 1 (axis aligned boxes); not behind holds
    # and the object centre is not behind: 20 - (s_j + RA2C) >= -1.3
    exp = (((s + RA2C - 20).abs() <= HALF_LEN + 2.0) & (20 - (s + RA2C) >= RT.BEHIND_M))[:, None] \
        & (d.abs() <= HALF_WID + 1.0)[None, :]
    assert torch.equal(blk[0, :, :, 0], exp)
    assert blk.all(-1).eq(exp).all()                               # static: every bin
    assert int(exp.sum()) == 6 * 7                                 # stations 14..19 (s 14.5 .. 19.5), |d| <= 1.8


def test_not_behind_and_absent():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    box = _static(10.0, 0.0, L=2.0, W=1.0)
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    with_b = RT.blocked_field(g, box[None], on)
    no_b = RT.blocked_field(g, box[None], on, behind_m=None)
    # cell rear axle at s: ego centre s + 1.461; the object is 'behind' when 10 - (s + 1.461) < -1.3 (s > 9.839)
    s = g.s[0]
    behind = (10.0 - (s + RA2C)) < RT.BEHIND_M
    hit_any = no_b[0, :, 8, 0]
    assert bool((hit_any & behind).any())                          # some overlapping cells have the object behind
    assert not bool((with_b[0, :, 8, 0] & behind).any())
    assert torch.equal(with_b[0, :, 8, 0], hit_any & ~behind)
    # ABSENT samples never block; int8 state accepted (OBS = 1)
    st = torch.zeros(1, 1, 50, dtype=torch.int8)
    st[..., 10:13] = 1                                              # t = 1.0 .. 1.2 s -> bin 4
    b2 = RT.blocked_field(g, box[None], st, behind_m=None)
    assert b2[..., 4].any() and not b2[..., :4].any() and not b2[..., 5:].any()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_equals_brute_force_and_numpy(seed):
    par, boxes, obs, tt = _random_scene(seed)
    g = corridor_geometry(par)
    fr = RT.parent_frame(g)
    for tbin in (0.25, 0.1, 0.5):
        M = RT.n_bins_of(tbin)
        B = RT.blocked_field(g, boxes[None], obs[None], tt, tbin_s=tbin, details=True)
        ref = _brute(fr, boxes, obs, tt, tbin, M)
        assert torch.equal(B["blocked"], ref), (tbin, int((B["blocked"] != ref).sum()))
        Bn = RT.blocked_field_np(RT.frame_np(fr), boxes.numpy(), obs.numpy(), tt.numpy(), tbin_s=tbin)
        assert np.array_equal(Bn, B["blocked"].numpy())
        # pooling: bin m = OR of its 0.1 s samples
        mt = torch.as_tensor(RT.time_bins(tt, tbin))
        for m in range(M):
            assert torch.equal(B["blocked"][..., m], B["fine"][..., mt == m].any(-1))
    assert int(ref.sum()) > 0
    # float32 geometry (training dtype): identical up to SAT ties at float rounding
    B32 = RT.blocked_field(corridor_geometry(par.float()), boxes.float()[None], obs[None], tt)
    B64 = RT.blocked_field(g, boxes[None], obs[None], tt)
    assert int((B32 != B64).sum()) <= 2


@pytest.mark.parametrize("seed", [0, 3])
def test_lon_max_field(seed):
    par, boxes, obs, tt = _random_scene(seed)
    g = corridor_geometry(par)
    fr = RT.parent_frame(g)
    B0 = RT.blocked_field(g, boxes[None], obs[None], tt)
    B, L = RT.blocked_field(g, boxes[None], obs[None], tt, lon_max=True)
    assert torch.equal(B, B0) and L.dtype == F64 and L.shape == B.shape        # opt-in: B unchanged
    assert torch.equal(L >= RT.BEHIND_M, B)                                      # B is a threshold of LON
    Bnb, Lnb = RT.blocked_field(g, boxes[None], obs[None], tt, behind_m=None, lon_max=True)
    assert torch.equal(Lnb > -math.inf, Bnb) and torch.equal(Lnb, L)           # LON has no not-behind filter
    _, Lref = _brute(fr, boxes, obs, tt, 0.25, 20, lon_max=True)
    assert torch.equal(torch.isfinite(L), torch.isfinite(Lref))
    fin = torch.isfinite(L)
    assert bool(fin.any()) and float((L[fin] - Lref[fin]).abs().max()) < 1e-12
    assert bool(((L < RT.BEHIND_M) & fin).any())                               # some overlaps are 'behind'
    d = RT.blocked_field(g, boxes[None], obs[None], tt, lon_max=True, details=True)
    assert torch.equal(d["lon_max"], L) and torch.equal(d["blocked"], B)
    Bn, Ln = RT.blocked_field_np(RT.frame_np(fr), boxes.numpy(), obs.numpy(), tt.numpy(), lon_max=True)
    assert np.array_equal(Bn, B.numpy()) and np.array_equal(np.isfinite(Ln), fin.numpy())
    assert np.abs(Ln[fin.numpy()] - L.numpy()[fin.numpy()]).max() < 1e-12
    Bn2, Fn2, Ln2 = RT.blocked_field_np(RT.frame_np(fr), boxes.numpy(), obs.numpy(), tt.numpy(), fine=True,
                                        lon_max=True)
    assert np.array_equal(Bn2, Bn) and np.array_equal(Ln2, Ln) and Fn2.shape == (3, 48, 17, 50)
    # float32 (training dtype): B is still exactly LON >= BEHIND_M
    B32, L32 = RT.blocked_field(corridor_geometry(par.float()), boxes.float()[None], obs[None], tt, lon_max=True)
    assert L32.dtype == torch.float32 and torch.equal(L32 >= RT.BEHIND_M, B32)


def test_heading_draft_and_index_and_mask():
    par, boxes, obs, tt = _random_scene(5, P=2, A=10)
    g = corridor_geometry(par)
    Bd = RT.blocked_field(g, boxes[None], obs[None], tt, heading="draft")
    ref = _brute(RT.parent_frame(g, "draft"), boxes, obs, tt, 0.25, 20)
    assert torch.equal(Bd, ref)
    # two scenes, parent -> scene index; empty second scene
    two = torch.stack([boxes, boxes + 100.0])
    Bi = RT.blocked_field(g, two, torch.stack([obs, obs]), tt, index=torch.tensor([1, 0]))
    B0 = RT.blocked_field(g, boxes[None], obs[None], tt)
    assert not Bi[0].any() and torch.equal(Bi[1], B0[1])
    # object mask removes everything
    Bm = RT.blocked_field(g, boxes[None], obs[None], tt, obj_mask=torch.zeros(1, boxes.shape[0], dtype=torch.bool))
    assert not Bm.any()
    with pytest.raises(ValueError):
        RT.blocked_field(g, torch.stack([boxes] * 3), torch.stack([obs] * 3), tt)   # 3 scenes, 2 parents, no index


def test_known_mask():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    B = RT.blocked_field(g, _static(20.0, 0.0)[None], torch.ones(1, 1, 50, dtype=torch.bool), details=True,
                         t_avail=torch.tensor([3.0]))
    kn = B["known"][0]
    assert kn[:12].all() and not kn[13:].any()                     # bin 12 = [3.0, 3.25): 3.1, 3.2 > t_avail
    assert not kn[12]


def test_gt_boxes_from_ref_matches_query():
    rng = np.random.default_rng(0)
    objs = []
    for A in (3, 5):
        kf = np.zeros((A, 11, 6), np.float32)
        kf[..., 0] = rng.normal(10, 5, (A, 1)) + np.arange(11) * rng.normal(0, 2, (A, 1))
        kf[..., 1] = rng.normal(0, 4, (A, 1))
        kf[..., 5] = 1
        meta = np.zeros((A, 5), np.int16)
        meta[:, GF.MT_LAST] = 10
        meta[:, GF.MT_AGENT] = 1
        first = np.zeros((A, 6), np.float32)
        first[:, 0], first[:, 1] = 4.0, 2.0
        objs.append(dict(kf=kf, first=first, meta=meta, n_kf=np.int32(11), R=np.float32(80)))
    pad = GF.pad_objects(objs)
    ref = {"ref_obj_kf": torch.as_tensor(pad["kf"]), "ref_obj_first": torch.as_tensor(pad["first"]),
           "ref_obj_meta": torch.as_tensor(pad["meta"]), "ref_obj_n": torch.tensor([3, 5]),
           "ref_obj_n_kf": torch.tensor([11, 9])}
    boxes, obs, agent, ta = RT.gt_boxes_from_ref(ref, torch.tensor([0, 1]))
    assert boxes.shape == (2, 5, 50, 5) and obs.shape == (2, 5, 50)
    b1, s1 = GF.query(dict(objs[1], n_kf=np.int32(9)), RT.fine_times(), dtype=np.float64)
    assert np.allclose(boxes[1].numpy(), b1, atol=1e-5) and np.array_equal(obs[1].numpy(), s1 == 1)
    assert not obs[0, 3:].any() and float(ta[1]) == 4.0 and bool(agent[0, :3].all()) and not agent[0, 3:].any()


# ----------------------------------------------------------------------------------------------- margins
def _road_sdf(fn):
    X, Y = grid_mesh()
    return torch.as_tensor(fn(X, Y), dtype=torch.float32)[None]     # [1, 320, 256]


def test_margins_straight_road_sdf_and_grid():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    asym = _road_sdf(lambda X, Y: np.minimum(4.0 - Y, Y + 1.5))                         # y in [-1.5, 4]
    ma = RT.margins(asym, g)
    assert torch.allclose(ma["m_left"][0], torch.full((48,), 4.0 - HALF_WID, dtype=F64), atol=2e-3)
    assert torch.allclose(ma["m_right"][0], torch.full((48,), 1.5 - HALF_WID, dtype=F64), atol=2e-3)
    assert bool(ma["valid"].all()) and not bool(ma["L_exit"][0]) and float(ma["L"][0]) == 48.0
    assert not bool(ma["m_unread_left"].any()) and not bool(ma["m_unread_right"].any())     # real boundaries
    sdf = _road_sdf(lambda X, Y: np.minimum(3.0 - np.abs(Y), 30.2 - X))                 # |y| <= 3, ends at x = 30.2
    m = RT.margins(sdf, g)
    ok = g.s[0] + RA2C + HALF_LEN < 30.2 - 2.0                             # stations whose box is well before the end
    assert torch.allclose(m["m_left"][0][ok], torch.full_like(m["m_left"][0][ok], 3.0 - HALF_WID), atol=2e-3)
    assert torch.allclose(m["m_right"][0][ok], torch.full_like(m["m_left"][0][ok], 3.0 - HALF_WID), atol=2e-3)
    assert bool(m["L_exit"][0]) and abs(float(m["L"][0]) - (30.2 - RA2C - HALF_LEN)) < 2e-3
    assert int(m["L_station"][0]) == int(math.floor(30.2 - RA2C - HALF_LEN - 0.5)) + 1
    # past the end the centred box is outside: margins = its (negative) min corner value
    j = int(m["L_station"][0]) + 2
    assert float(m["m_left"][0, j]) < 0 and float(m["m_left"][0, j]) == float(m["m_right"][0, j])
    # grid version: D = the SDF at the grid points, same tube (lateral field linear -> bilinear exact)
    D, Dv = sample_sdf(sdf, g.points.float().reshape(1, -1, 2))
    mg = RT.margins(D.reshape(1, 48, 17).double(), g)
    assert torch.allclose(mg["m_left"][0][ok], m["m_left"][0][ok], atol=2e-3)
    assert torch.allclose(mg["m_right"][0][ok], m["m_right"][0][ok], atol=2e-3)
    assert abs(float(mg["L"][0]) - float(m["L"][0])) < 2e-2


def test_margins_capped_and_curved_ring_road():
    # wide road: no crossing within 4.8 m -> capped (sdf); grid version capped by the grid's lateral reach
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    wide = _road_sdf(lambda X, Y: 9.0 - np.abs(Y))
    m = RT.margins(wide, g)
    assert bool(m["m_capped_left"][0, :30].all())
    assert torch.allclose(m["m_left"][0, :30], torch.full((30,), 4.8, dtype=F64))
    assert not bool(m["L_exit"][0]) and abs(float(m["L"][0]) - 48.0) < 1e-9
    D, _ = sample_sdf(wide, g.points.float().reshape(1, -1, 2))
    mg = RT.margins(D.reshape(1, 48, 17).double(), g)
    assert torch.allclose(mg["m_left"][0, :30], torch.full((30,), 4.8 - HALF_WID, dtype=F64), atol=0.051)
    # ... and that limit is flagged as the grid's edge (unread), not a curb; the sdf scan is capped, not unread
    assert bool(mg["m_unread_left"][0, :30].all()) and bool(mg["m_unread_right"][0, :30].all())
    assert not bool(mg["m_capped_left"][0, :30].any())
    assert bool(m["m_capped_left"][0, :30].all()) and not bool(m["m_unread_left"][0, :30].any())
    # ring road of radius 25 m (left turn), lane |r - 25| <= 3: margins from the exact box corners
    k = 1.0 / 25.0
    g2 = corridor_geometry(_draft(6.0, k)[None])
    ring = _road_sdf(lambda X, Y: 3.0 - np.abs(np.hypot(X, Y - 25.0) - 25.0))
    m2 = RT.margins(ring, g2)
    # analytic: the box at shift delta (radius 25 - delta) has its binding corner on the outer (front-right / rear
    # right) or inner side; check numerically the scan's crossing is the exact boundary of min corner sdf = 0
    fr = RT.parent_frame(g2)
    for j in (3, 10, 20):
        for side, sg in (("m_left", 1.0), ("m_right", -1.0)):
            dl = float(m2[side][0, j])
            for eps, inside in ((-0.02, True), (0.02, False)):
                ra = fr["p"][0, j] + sg * (dl + eps) * fr["n"][0, j]
                pose = torch.cat([ra, fr["h"][0, j:j + 1]])[None, None]
                v, _ = sample_sdf(ring, ego_corners(pose).reshape(1, 4, 2))
                assert (bool((v > 0).all()) == inside), (j, side, eps, v)
    D2, _ = sample_sdf(ring, g2.points.float().reshape(1, -1, 2))
    mg2 = RT.margins(D2.reshape(1, 48, 17).double(), g2)
    sel = slice(2, 30)
    assert float((mg2["m_left"][0, sel] - m2["m_left"][0, sel]).abs().max()) < 0.06
    assert float((mg2["m_right"][0, sel] - m2["m_right"][0, sel]).abs().max()) < 0.06


# ----------------------------------------------------------------------------------------------- v_max
def test_v_max_inverts_track_table():
    kap = torch.tensor([0.0, 0.01, 0.05, 0.2, -0.05], dtype=F64)
    for mo in (0.2, 0.33, 0.6, 1.0, 2.5):
        m_out = torch.full_like(kap, mo)
        v = RT.v_max(kap, m_out)
        if mo - RT.TRACK_MARGIN < RT.TRACK_ERR_MEDIAN[0]:
            assert (v == 0).all()
            continue
        assert float(v[0]) == RT.V_CAP
        for i in range(1, 5):
            if float(v[i]) < RT.V_CAP:
                e = RT.track_error(v[i] * kap[i].abs())
                assert abs(float(e) + RT.TRACK_MARGIN - mo) < 1e-9
    # monotone in the margin, decreasing in |kappa|; comfort prior only lowers
    mm = torch.linspace(0.3, 3.0, 50, dtype=F64)
    v1 = RT.v_max(torch.full_like(mm, 0.05), mm)
    assert bool((v1[1:] >= v1[:-1]).all())
    vk = RT.v_max(torch.tensor([0.02, 0.05, 0.1], dtype=F64), torch.full((3,), 1.0, dtype=F64))
    assert bool((vk[1:] < vk[:-1]).all())
    vc = RT.v_max(torch.tensor([0.02, 0.05, 0.1], dtype=F64), torch.full((3,), 1.0, dtype=F64), comfort_prior=True)
    assert bool((vc <= vk).all()) and float(vc[2]) <= math.sqrt(4.89 / 0.1) + 1e-9
    # outside margin
    om = RT.outside_margin(torch.tensor([0.1, -0.1, 0.0]), torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 0.5]))
    assert om.tolist() == [2.0, 1.0, 0.5]


def test_fit_track_table_synthetic():
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 0.69, 200000)
    e = 2.0 * w + rng.normal(0, 0.05, w.shape)
    xk, yk, n = RT.fit_track_table(w, e)
    assert len(xk) == len(RT.TRACK_BIN_EDGES) - 1 and (np.diff(yk) >= 0).all()
    assert np.allclose(yk, 2.0 * xk, atol=0.02)
    assert len(RT.TRACK_YAW) == len(RT.TRACK_ERR_MEDIAN) == len(RT.TRACK_ERR_P90)
    assert (np.diff(RT.TRACK_YAW) > 0).all() and (np.diff(RT.TRACK_ERR_MEDIAN) > 0).all()
    assert (np.diff(RT.TRACK_ERR_P90) > 0).all()


@pytest.mark.skipif(not os.path.isfile(FEATS), reason="F1 feats.parquet not available")
def test_track_table_provenance():
    import pyarrow.parquet as pq
    t = pq.read_table(FEATS, columns=["vtype", "max_yaw_rate", "max_path"])
    par = t.column("vtype").to_numpy(zero_copy_only=False) == "par"
    w = t.column("max_yaw_rate").to_numpy()[par]
    e = t.column("max_path").to_numpy()[par]
    xk, yk, n = RT.fit_track_table(w, e)
    assert np.allclose(xk, RT.TRACK_YAW, atol=6e-4) and np.allclose(yk, RT.TRACK_ERR_MEDIAN, atol=6e-4)
    _, y90, _ = RT.fit_track_table(w, e, q=90.0)
    assert np.allclose(y90, RT.TRACK_ERR_P90, atol=6e-4)
    assert int(n.sum()) == 96000 - int((w >= 0.7).sum()) - int((~np.isfinite(w)).sum())


# ----------------------------------------------------------------------------------------------- windows
def test_windows():
    B = torch.zeros(1, 48, 17, 20, dtype=torch.bool)
    B[0, 0, 1, 2:4] = True                       # [0.5, 1.0)
    B[0, 0, 2, 19] = True                        # last bin
    B[0, 0, 3, 1] = True
    B[0, 0, 3, 5:7] = True                       # two runs, last ends 1.75
    W = RT.windows(B)
    assert math.isinf(float(W["t_in_block"][0, 0, 0])) and float(W["t_out_block"][0, 0, 0]) == 0.0
    assert float(W["t_in_block"][0, 0, 1]) == 0.5 and float(W["t_out_block"][0, 0, 1]) == 1.0
    assert float(W["t_in_block"][0, 0, 2]) == 4.75 and math.isinf(float(W["t_out_block"][0, 0, 2]))
    assert float(W["t_in_block"][0, 0, 3]) == 0.25 and float(W["t_out_block"][0, 0, 3]) == 1.75
    assert W["n_runs"][0, 0, :4].tolist() == [0, 1, 1, 2]
    kn = torch.ones(1, 20, dtype=torch.bool)
    kn[0, 6:] = False                            # last known bin 5 blocked at cell 3 -> +inf; cell 2 blocked unknown
    W2 = RT.windows(B, known=kn)
    assert math.isinf(float(W2["t_out_block"][0, 0, 3])) and float(W2["t_out_block"][0, 0, 2]) == 0.0
    # a parent with no known bin is unknown, not free: blocked from t0, never free (parent 1 unaffected)
    B2 = torch.cat([B, B])
    kn2 = torch.stack([torch.zeros(20, dtype=torch.bool), torch.ones(20, dtype=torch.bool)])
    W3 = RT.windows(B2, known=kn2)
    assert bool(torch.isinf(W3["t_out_block"][0]).all()) and bool((W3["t_in_block"][0] == 0).all())
    assert int(W3["n_runs"][0].sum()) == 0
    for k in W:
        assert torch.equal(W3[k][1], W[k][0])


# ----------------------------------------------------------------------------------------------- executed slack
def _states(v, n=41, y=0.0, x0=0.0):
    t = torch.arange(n, dtype=F64) * 0.1
    x = x0 + v * t
    return torch.stack([x, torch.full_like(t, y), torch.zeros_like(t), torch.full_like(t, v), 0 * t, 0 * t], -1)


def test_executed_slack_nc_static_and_crossing():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    B = RT.blocked_field(g, _static(20.0, 0.0)[None], on)
    st = _states(5.0)[None]
    out = RT.executed_slack(st, g, B)
    assert bool(out["hit_nc"][0]) and float(out["slack_nc"][0]) == -RT.SLACK_CAP       # static: blocked to the end
    # critical state: first state whose nearest cell is blocked (station 14: s in [14, 15))
    assert int(out["crit_station"][0, 0]) == 14 and abs(float(out["crit_time"][0, 0]) - 2.8) < 1e-9
    # exact executed boxes first overlap later than the cell read (cell = station centre, up to ds / 2 ahead)
    gaps = box_gaps(st[0, :, :3], _static(20.0, 0.0)[0, 0]).max(-1).values
    n_ex = int(torch.nonzero(gaps <= 0)[0])
    assert abs(n_ex * 0.1 - float(out["crit_time"][0, 0])) <= 0.5 / 5.0 + 1e-9
    # crossing object that has left before the ego arrives: positive slack = time distance to the window
    T = 50
    tt = torch.arange(T, dtype=F64) * 0.1
    box = torch.stack([torch.full_like(tt, 20.0), -10.0 + 20.0 * tt, torch.full_like(tt, math.pi / 2),
                       torch.full_like(tt, 4.0), torch.full_like(tt, 2.0)], -1)[None]    # crosses y = 0 at t = 0.5 s
    B2 = RT.blocked_field(g, box[None], on)
    o2 = RT.executed_slack(st, g, B2)
    assert not bool(o2["hit_nc"][0]) and 0 < float(o2["slack_nc"][0]) < RT.SLACK_CAP
    # exact: the object blocks stations 15..19 (|s_j + RA2C - 20| <= 2.588 + 1, not behind) on the centre lane while
    # |y_obj| <= 1.1485 + 2 (t in [0.343, 0.657]: samples 0.4 .. 0.6 -> bins 1, 2 = run [0.25, 0.75)); the ego reaches
    # station 15 at t = 3.0 -> slack = t - end of the run = 2.25
    assert torch.nonzero(B2[0, :, 8].any(-1)).flatten().tolist() == [15, 16, 17, 18, 19]
    assert torch.nonzero(B2[0, 15, 8]).flatten().tolist() == [1, 2]
    assert abs(float(o2["slack_nc"][0]) - 2.25) < 1e-9 and int(o2["crit_station"][0, 0]) == 15
    # the same crossing at t = 4.5 (run [4.25, 4.75)): the ego has left station 19 at 3.9 s -> slack = 4.25 - 3.9
    late = box.clone()
    late[..., 1] = 20.0 * (tt - 4.5)
    o_l = RT.executed_slack(st, g, RT.blocked_field(g, late[None], on))
    assert not bool(o_l["hit_nc"][0]) and abs(float(o_l["slack_nc"][0]) - 0.35) < 1e-9
    assert int(o_l["crit_station"][0, 0]) == 19 and abs(float(o_l["crit_time"][0, 0]) - 3.9) < 1e-9
    # an ego already near the crossing meets it -> hit, slack = t - (end of the blocked run of that cell)
    near = RT.executed_slack(_states(5.0, x0=15.0)[None], g, B2)
    assert bool(near["hit_nc"][0]) and abs(float(near["slack_nc"][0]) - (0.3 - 0.75)) < 1e-9
    # conservative read: the 4 surrounding grid points -> enters the block no later than the nearest-cell read
    o4 = RT.executed_slack(st, g, B, read="any4")
    assert bool(o4["hit_nc"][0]) and float(o4["crit_time"][0, 0]) <= float(out["crit_time"][0, 0])
    o4c = RT.executed_slack(st, g, B2, read="any4")
    assert float(o4c["slack_nc"][0]) <= float(o2["slack_nc"][0])
    # never blocked -> cap, no critical station
    o3 = RT.executed_slack(st, g, torch.zeros_like(B))
    assert float(o3["slack_nc"][0]) == RT.SLACK_CAP and int(o3["crit_station"][0, 0]) == -1


def test_executed_slack_stopped_and_ttc():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    B = RT.blocked_field(g, _static(20.0, 0.0)[None], on)
    # stopped ego inside a blocked cell: NC skipped (v <= 0.05), TTC skipped (v < 5e-3)
    st = _states(0.0, x0=15.0)[None]
    o = RT.executed_slack(st, g, B)
    assert not bool(o["hit_nc"][0]) and not bool(o["hit_ttc"][0])
    o_off = RT.executed_slack(st, g, B, stop_nc=None, stop_ttc=None)
    assert bool(o_off["hit_nc"][0]) and bool(o_off["hit_ttc"][0])
    # TTC: an ego that stops just before the box -- current boxes free, the 0.9 s projection enters the blocked cells
    t = torch.arange(41, dtype=F64) * 0.1
    tc = torch.clamp(t, max=2.0)
    v = 6.0 - 3.0 * tc                                                                      # -3 m/s^2, stops at 2 s
    x = 6.0 * tc - 1.5 * tc ** 2                                                            # at x = 6 m (station 6)
    st2 = torch.stack([x, 0 * t, 0 * t, v, 0 * t, 0 * t], -1)[None]
    B2 = RT.blocked_field(g, _static(13.0, 0.0)[None], on)                                 # blocks stations 7..12
    assert not B2[0, :7].any() and B2[0, 7, 8].all()
    o2 = RT.executed_slack(st2, g, B2)
    # max of x + 0.9 v = 7.215 m at t = 1.1 s: the 0.9 s projection reaches station 7 for t in [0.72, 1.48]
    assert not bool(o2["hit_nc"][0]) and bool(o2["hit_ttc"][0])
    assert abs(float(o2["crit_time"][0, 1]) - 0.8) < 1e-9 and int(o2["crit_station"][0, 1]) == 7


def test_executed_slack_dac_and_index():
    g = corridor_geometry(torch.stack([_draft(5.0, 0.0), _draft(5.0, 0.0)]))
    sdf = _road_sdf(lambda X, Y: np.minimum(4.0 - Y, Y + 1.5))
    st = torch.stack([_states(5.0, y=0.0), _states(5.0, y=-0.3), _states(5.0, y=2.0)])
    B = torch.zeros(2, 48, 17, 20, dtype=torch.bool)
    o = RT.executed_slack(st, g, B, field=sdf, index=torch.tensor([0, 1, 1]),
                          sdf_index=torch.zeros(3, dtype=torch.long))
    corners = ego_corners(st[..., :3])
    v, _ = sample_sdf(sdf, corners.reshape(3, -1, 2), torch.zeros(3, dtype=torch.long))
    assert torch.allclose(o["slack_dac"], v.min(-1).values.to(F64), atol=1e-6)
    assert o["hit_dac"].tolist() == [False, False, False]
    st_out = _states(5.0, y=-1.0)[None]                                                      # right corners at -2.15
    o_out = RT.executed_slack(st_out, g, B, field=sdf, index=torch.tensor([0]))
    assert bool(o_out["hit_dac"][0]) and abs(float(o_out["slack_dac"][0]) - (1.5 - 1.0 - HALF_WID)) < 1e-4
    # grid version on D = SDF at the grid points agrees on this linear road
    D, _ = sample_sdf(sdf, g.points.float().reshape(1, -1, 2))
    og = RT.executed_slack(st, g, B, field=D.reshape(2, 48, 17).double(), index=torch.tensor([0, 1, 1]))
    assert torch.allclose(og["slack_dac"], o["slack_dac"], atol=2e-3)


def test_ttc_not_behind_from_current_ego():
    """official TTC: is_agent_ahead / is_agent_behind from the CURRENT rear axle, not from the projected box."""
    g = corridor_geometry(_draft(5.0, 0.0)[None])                  # straight, ds = 1 m
    # a small box (1 x 1 m) at x = 8.5, observed only at t = 0.8 / 0.9 s (bin 3); ego state 0 at x = 0 with 10 m/s,
    # the other states stopped (no NC / TTC reads): only the 0.9 s projection (rear axle at 9 m, read at t = 0.9)
    # reaches it, with the box in its REAR half
    box = _box(8.5, 0.0, 0.0, L=1.0, W=1.0)
    obs = torch.zeros(1, 1, 50, dtype=torch.bool)
    obs[..., 8:10] = True
    B, L = RT.blocked_field(g, box[None], obs, lon_max=True)
    st = torch.zeros(1, 41, 6, dtype=F64)
    st[0, 0, 3] = 10.0
    # exact boxes: the projected ego box overlaps the object; it is 'behind' from the projected box centre
    # (8.5 - 10.461 < -1.3) but ahead of the current ego centre (8.5 - 1.461) -> the official rule flags it
    pr = torch.tensor([9.0, 0.0, 0.0], dtype=F64)
    assert float(box_gaps(pr, box[0, 9]).max()) <= 0
    assert 8.5 - (9.0 + RA2C) < RT.BEHIND_M <= 8.5 - RA2C
    # field: the projected cell (station 9) holds the object behind its box centre -> B misses it ...
    assert not bool(B[0, 9, 8, 3]) and abs(float(L[0, 9, 8, 3]) - (8.5 - 9.5 - RA2C)) < 1e-12
    o0 = RT.executed_slack(st, g, B)
    assert not bool(o0["hit_ttc"][0]) and not bool(o0["hit_nc"][0])
    # ... the LON field re-reads not-behind from the current ego: LON + |v| delta >= BEHIND_M
    o1 = RT.executed_slack(st, g, B, lon_field=L)
    assert bool(o1["hit_ttc"][0]) and not bool(o1["hit_nc"][0])
    assert int(o1["crit_station"][0, 1]) == 9 and float(o1["crit_time"][0, 1]) == 0.0
    assert abs(float(o1["slack_ttc"][0]) - (0.9 - 1.0)) < 1e-9          # read 0.9 s inside the run [0.75, 1.0)
    o2 = RT.executed_slack(st, g, None, lon_field=L)                     # B derived from LON
    assert all(torch.equal(o1[k], o2[k]) for k in o1)
    # delta 0 only: the LON read reduces to B; NC never changes
    par, boxes, obs_r, tt = _random_scene(4)
    gr = corridor_geometry(par)
    Br, Lr = RT.blocked_field(gr, boxes[None], obs_r[None], tt, lon_max=True)
    x = torch.arange(41, dtype=F64) * 0.1 * 8.0
    sr = torch.stack([torch.stack([x, 0 * x + 0.3 * k, 0 * x, 0 * x + 8.0, 0 * x, 0 * x], -1) for k in range(3)])
    a = RT.executed_slack(sr, gr, Br, deltas=(0.0,), per_state=True)
    b = RT.executed_slack(sr, gr, Br, deltas=(0.0,), per_state=True, lon_field=Lr)
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = RT.executed_slack(sr, gr, Br, per_state=True)
    d = RT.executed_slack(sr, gr, Br, per_state=True, lon_field=Lr)
    assert torch.equal(c["state_nc"], d["state_nc"]) and torch.equal(c["hit_nc"], d["hit_nc"])
    assert bool((d["state_ttc"] <= c["state_ttc"]).all())                # the current-ego rule only adds hits here
    d4 = RT.executed_slack(sr, gr, Br, lon_field=Lr, read="any4")
    assert bool((d4["slack_ttc"] <= d["slack_ttc"]).all())


def _line_states(fr, x, v):
    """states [1, 41, 6] at arc lengths x [41] on the straight frame's centreline, heading along it, speed v."""
    o, e = fr["p"][0, 0] - fr["s"][0, 0] * fr["u"][0, 0], fr["u"][0, 0]
    h = torch.full_like(x, float(fr["h"][0, 0]))
    return torch.cat([o + x[:, None] * e, h[:, None], v[:, None], torch.zeros(41, 2, dtype=F64)], -1)[None]


def test_executed_slack_heading_30deg():
    th = math.radians(30.0)
    fr = _line_frame(th, 2.0, -1.0)
    e = fr["u"][0, 0]
    o = torch.tensor([2.0, -1.0], dtype=F64)
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    t = torch.arange(41, dtype=F64) * 0.1
    # NC: a 4 x 2 m box 20 m along the line, aligned -> stations 14..19 (as at heading 0), lateral |d| <= 1.8 m;
    # ego at 4.7 m/s: s = 0.47 n first enters station 14 at n = 30 (s = 14.1)
    c = o + 20.0 * e
    B = RT.blocked_field(fr, _box(float(c[0]), float(c[1]), th)[None], on)
    assert torch.nonzero(B[0, :, 8, 0]).flatten().tolist() == list(range(14, 20))
    assert torch.nonzero(B[0, 16, :, 0]).flatten().tolist() == list(range(5, 12))
    out = RT.executed_slack(_line_states(fr, 4.7 * t, torch.full_like(t, 4.7)), fr, B)
    assert bool(out["hit_nc"][0]) and int(out["crit_station"][0, 0]) == 14
    assert abs(float(out["crit_time"][0, 0]) - 3.0) < 1e-9
    # TTC: the braking ego of test_executed_slack_stopped_and_ttc rotated by 30 deg (box at 13 m: stations 7..12);
    # max x + 0.9 v = 7.215 at 1.1 s, the 0.9 s projection first reaches station 7 at t = 0.8 (7.08 m).  A wrong sign
    # of sin(h) in the projection would leave the line by 0.9 v sin(60 deg) and never be blocked
    tc = torch.clamp(t, max=2.0)
    c2 = o + 13.0 * e
    B2 = RT.blocked_field(fr, _box(float(c2[0]), float(c2[1]), th)[None], on)
    o2 = RT.executed_slack(_line_states(fr, 6.0 * tc - 1.5 * tc ** 2, 6.0 - 3.0 * tc), fr, B2)
    assert not bool(o2["hit_nc"][0]) and bool(o2["hit_ttc"][0])
    assert int(o2["crit_station"][0, 1]) == 7 and abs(float(o2["crit_time"][0, 1]) - 0.8) < 1e-9


def test_executed_slack_ring_exact_geometry():
    """left-turning ring (kappa 0.04): the TTC projections leave the path straight along the executed heading (outward,
    d < 0); expected cells from the exact polar geometry (s = R angle, d = R - |q - c|) at every (state, delta)."""
    R = 25.0
    fr = _ring_frame(R)
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    t = torch.arange(41, dtype=F64) * 0.1

    def ring_states(s, v):
        ph = s / R
        return torch.stack([R * torch.sin(ph), R * (1 - torch.cos(ph)), ph, v, 0 * s, 0 * s], -1)[None]

    def ring_box(s):
        return _box(R * math.sin(s / R), R * (1 - math.cos(s / R)), s / R)

    # NC: box on the ring at arc 20 -> stations 14..19; the ego at 4.7 m/s enters station 14 at t = 3.0
    B = RT.blocked_field(fr, ring_box(20.0)[None], on)
    assert torch.nonzero(B[0, :, 8, 0]).flatten().tolist() == list(range(14, 20))
    out = RT.executed_slack(ring_states(4.7 * t, torch.full_like(t, 4.7)), fr, B)
    assert bool(out["hit_nc"][0]) and int(out["crit_station"][0, 0]) == 14
    assert abs(float(out["crit_time"][0, 0]) - 3.0) < 1e-9
    # TTC: braking ego on the ring, box at arc 13
    tc = torch.clamp(t, max=2.0)
    v = 6.0 - 3.0 * tc
    st = ring_states(6.0 * tc - 1.5 * tc ** 2, v)
    B2 = RT.blocked_field(fr, ring_box(13.0)[None], on)
    o2 = RT.executed_slack(st, fr, B2, per_state=True)
    exp = torch.zeros(41, 4, dtype=torch.bool)
    first = None
    for n in range(41):
        for k, dl in enumerate(RT.TTC_DELTAS):
            q = st[0, n, :2] + v[n] * dl * torch.stack([torch.cos(st[0, n, 2]), torch.sin(st[0, n, 2])])
            rel = q - torch.tensor([0.0, R], dtype=F64)
            sq, dq = R * math.atan2(float(rel[0]), float(-rel[1])), R - float(rel.norm())
            j, i = math.floor(sq), round((dq + 4.8) / 0.6)
            exp[n, k] = bool(v[n] >= RT.STOP_TTC) and 0 <= j < 48 and bool(B2[0, j, i, 0])
            if exp[n, k] and first is None:
                first = (n, j, i, dq)
    assert torch.equal(o2["state_ttc"][0] < 0, exp) and int(exp.sum()) > 0
    assert not bool(o2["hit_nc"][0]) and bool(o2["hit_ttc"][0])
    assert first[2] == 8 and first[3] < -0.2                            # the first flagged projection is outward
    assert int(o2["crit_station"][0, 1]) == first[1] == 7 and abs(float(o2["crit_time"][0, 1]) - 0.1 * first[0]) < 1e-9
    assert first[0] == 8


@pytest.mark.parametrize("tbin", [0.1, 0.25, 0.5])
def test_executed_slack_float32_bins(tbin):
    """float32 (training dtype) = float64: read times are binned in float64 (t / 0.1 rounds below 37 / 42 / 47 in
    float32 for t = 3.7 / 4.2 / 4.7 s)."""
    tn32 = torch.arange(41, dtype=torch.float32) * 0.1
    tt32 = (tn32[:, None] + torch.tensor(RT.TTC_DELTAS, dtype=torch.float32)[None]).reshape(-1)
    tt64 = (torch.arange(41, dtype=F64)[:, None] * 0.1 + torch.tensor(RT.TTC_DELTAS, dtype=F64)[None]).reshape(-1)
    if tbin == 0.1:                                                      # the float32 trap the module avoids
        assert int((torch.floor(tt32 / tbin + RT.T_EPS) != torch.floor(tt64 / tbin + RT.T_EPS)).sum()) == 3
    M = RT.n_bins_of(tbin)
    g = torch.Generator().manual_seed(7)
    B = torch.zeros(1, 48, 17, M, dtype=torch.bool)
    B[0, 9, 8] = torch.rand(M, generator=g) < 0.5                        # one cell, random bins
    fr64 = _line_frame(0.0)
    fr32 = {k: v.float() for k, v in fr64.items()}
    # every state at station 9's centre with 0.06 m/s (> the stopped-ego thresholds; projections stay in the cell)
    st = torch.stack([torch.full((41,), 9.5, dtype=F64), torch.zeros(41, dtype=F64), torch.zeros(41, dtype=F64),
                      torch.full((41,), 0.06, dtype=F64), torch.zeros(41, dtype=F64), torch.zeros(41, dtype=F64)], -1)
    a = RT.executed_slack(st[None], fr64, B, tbin_s=tbin, per_state=True)
    b = RT.executed_slack(st[None].float(), fr32, B, tbin_s=tbin, per_state=True)
    for k in ("hit_nc", "hit_ttc", "crit_station"):
        assert torch.equal(a[k], b[k]), k
    for k in ("state_nc", "state_ttc", "slack_nc", "slack_ttc", "crit_time"):
        assert torch.allclose(a[k].float(), b[k], atol=1e-5), k
    assert torch.equal(a["state_ttc"] < 0, B[0, 9, 8][torch.floor(tt64 / tbin + RT.T_EPS).long().clamp(max=M - 1)]
                       .reshape(1, 41, 4) & (tt64 < M * tbin).reshape(1, 41, 4))


def test_executed_slack_dac_critical_state_and_speed_sign():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    sdf = _road_sdf(lambda X, Y: np.minimum(4.0 - Y, Y + 1.5))                          # y in [-1.5, 4]
    t = torch.arange(41, dtype=F64) * 0.1
    x = 5.0 * t
    B = torch.zeros(1, 48, 17, 20, dtype=torch.bool)

    def st(xx, yy, vv):
        return torch.stack([xx, yy, 0 * xx, vv, 0 * xx, 0 * xx], -1)[None]

    # deepest DAC state on the grid: its station; off the grid (|d| > 5.1 m): station -1, time still the state's
    y = torch.zeros(41, dtype=F64)
    y[10:] = -1.0
    y[12] = -1.2
    o = RT.executed_slack(st(x, y, torch.full_like(x, 5.0)), g, B, field=sdf)
    assert bool(o["hit_dac"][0]) and int(o["crit_station"][0, 2]) == 6 and abs(float(o["crit_time"][0, 2]) - 1.2) < 1e-9
    y = torch.zeros(41, dtype=F64)
    y[20:] = -6.0
    y[25] = -6.5
    o = RT.executed_slack(st(x, y, torch.full_like(x, 5.0)), g, B, field=sdf)
    assert bool(o["hit_dac"][0]) and int(o["n_oog"][0]) == 21
    assert int(o["crit_station"][0, 2]) == -1 and abs(float(o["crit_time"][0, 2]) - 2.5) < 1e-9
    # no readable corner: slack = SDF cap, no critical state (sdf: outside the E grid; grid D: |d| > 4.8 m)
    o = RT.executed_slack(st(x + 500.0, 0 * x, torch.full_like(x, 5.0)), g, B, field=sdf, per_state=True)
    assert not bool(o["hit_dac"][0]) and float(o["slack_dac"][0]) == RT.SDF_CLIP
    assert int(o["crit_station"][0, 2]) == -1 and float(o["crit_time"][0, 2]) == -1.0
    assert bool((o["state_dac"] == RT.SDF_CLIP).all())
    D, _ = sample_sdf(sdf, g.points.float().reshape(1, -1, 2))
    o = RT.executed_slack(st(x, 0 * x + 20.0, torch.full_like(x, 5.0)), g, B, field=D.reshape(1, 48, 17).double())
    assert not bool(o["hit_dac"][0]) and float(o["slack_dac"][0]) == RT.SDF_CLIP
    assert int(o["crit_station"][0, 2]) == -1 and float(o["crit_time"][0, 2]) == -1.0
    # speed = |v| (official hypot): a slowly reversing state inside a blocked cell counts for NC and TTC ...
    on = torch.ones(1, 1, 50, dtype=torch.bool)
    Bs = RT.blocked_field(g, _static(20.0, 0.0)[None], on)
    xs = torch.full((41,), 15.0, dtype=F64)
    o = RT.executed_slack(st(xs, 0 * xs, torch.full_like(xs, -0.1)), g, Bs)
    assert bool(o["hit_nc"][0]) and bool(o["hit_ttc"][0]) and float(o["crit_time"][0, 0]) == 0.0
    # ... and is projected FORWARD along its heading: reversing at 3 m/s from x = 0, box at 8 m (stations 2..7):
    # the 0.9 s projection (rear axle 2.7 m, station 2) is blocked; the current box (station 0) is not
    B8 = RT.blocked_field(g, _static(8.0, 0.0)[None], on)
    assert torch.nonzero(B8[0, :, 8, 0]).flatten().tolist() == list(range(2, 8))
    vv = torch.zeros(41, dtype=F64)
    vv[0] = -3.0
    o = RT.executed_slack(st(0 * x, 0 * x, vv), g, B8)
    assert not bool(o["hit_nc"][0]) and bool(o["hit_ttc"][0])
    assert int(o["crit_station"][0, 1]) == 2 and float(o["crit_time"][0, 1]) == 0.0


def test_executed_slack_probability_field_and_late_reads():
    g = corridor_geometry(_draft(5.0, 0.0)[None])
    Bs = RT.blocked_field(g, _static(20.0, 0.0)[None], torch.ones(1, 1, 50, dtype=torch.bool))
    st = _states(5.0)[None]
    a = RT.executed_slack(st, g, Bs)
    b = RT.executed_slack(st, g, torch.where(Bs, 0.7, 0.3))                  # probabilities: blocked where > 0.5
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):
        RT.executed_slack(st, g, None)
    # a read time beyond the last bin is free and sees the last bin (not the clamped one)
    prof = torch.tensor([[False, False, False, True]])
    assert abs(float(RT._time_slack(prof, torch.tensor([1.2], dtype=F64), 0.25, 5.0)[0]) - 0.2) < 1e-12
    assert abs(float(RT._time_slack(prof, torch.tensor([0.5], dtype=F64), 0.25, 5.0)[0]) - 0.25) < 1e-12
    assert float(RT._time_slack(prof, torch.tensor([0.8], dtype=F64), 0.25, 5.0)[0]) == -5.0
