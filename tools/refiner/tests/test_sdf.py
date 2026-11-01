"""Tests for navsim/agents/para_ssr/refiner/sdf.py (CPU, < 1 min).

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_sdf.py

Synthetic tests pin the grid / sign / lookup conventions analytically; the metric-cache tests (skipped when the
navtest cache is missing) check the field against the official DAC predicate and the official scorer on 4 tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.refiner import sdf as S  # noqa: E402

MC_ROOT = ROOT / "data/exp/metric_cache"
TABLE = ROOT / "report/perception_reliability/pdm_attr/table.parquet"


def _rng(seed=0):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------------------- grid / lookup
def test_grid_convention():
    xs, ys = S.grid_centres()
    assert xs.shape == (320,) and ys.shape == (256,)
    assert xs[0] == -7.875 and xs[-1] == 71.875          # row 0 = rear (x = -8..-7.75 cell)
    assert ys[0] == 31.875 and ys[-1] == -31.875         # col 0 = LEFT (y_left = +32)
    assert np.allclose(np.diff(xs), 0.25) and np.allclose(np.diff(ys), -0.25)


def test_sample_at_cell_centres_is_exact():
    f = _rng(1).normal(0, 3, (320, 256)).astype(np.float16)
    X, Y = S.grid_mesh()
    idx = _rng(2).integers(0, [320, 256], (500, 2))
    xy = np.stack([X[idx[:, 0], idx[:, 1]], Y[idx[:, 0], idx[:, 1]]], -1)
    v, m = S.sample_sdf(torch.from_numpy(f), torch.from_numpy(xy).float())
    assert m.all()
    assert torch.equal(v, torch.from_numpy(f[idx[:, 0], idx[:, 1]].astype(np.float32)))
    vn, mn = S.sample_sdf_np(f, xy)
    assert mn.all() and np.array_equal(vn, f[idx[:, 0], idx[:, 1]].astype(np.float64))


def test_torch_matches_numpy_and_grid_sample():
    f = _rng(3).normal(0, 3, (320, 256))
    xy = np.stack([_rng(4).uniform(-9, 73, 4000), _rng(5).uniform(-33, 33, 4000)], -1)
    v, m = S.sample_sdf(torch.from_numpy(f), torch.from_numpy(xy))            # float64 path
    vn, mn = S.sample_sdf_np(f, xy)
    assert np.array_equal(m.numpy(), mn)
    assert np.abs(v.numpy() - vn).max() < 1e-12
    # grid_sample(align_corners=False) on [1,1,H=x,W=y] with normalised (col, row) coordinates
    u = (xy[:, 0] - S.E_X0) / S.E_RES                    # in cell units, centre i at i + 0.5
    w = (S.E_Y0 - xy[:, 1]) / S.E_RES
    grid = torch.from_numpy(np.stack([2 * w / S.E_W - 1, 2 * u / S.E_H - 1], -1))[None, :, None, :]
    gs = torch.nn.functional.grid_sample(torch.from_numpy(f)[None, None], grid, mode="bilinear",
                                         padding_mode="border", align_corners=False)[0, 0, :, 0].numpy()
    assert np.abs(gs[mn] - vn[mn]).max() < 1e-9
    # float32 path
    v32, _ = S.sample_sdf(torch.from_numpy(f.astype(np.float32)), torch.from_numpy(xy).float())
    assert v32.dtype == torch.float32 and np.abs(v32.numpy() - vn).max() < 1e-3   # rough N(0,3) field


def test_out_of_grid_mask_value_and_gradient():
    f = torch.from_numpy(_rng(6).normal(0, 3, (320, 256)).astype(np.float32))
    e = 1e-3
    pts = torch.tensor([
        [S.X_MIN_VALID + e, 0.0], [S.X_MIN_VALID - e, 0.0], [S.X_MAX_VALID - e, 0.0], [S.X_MAX_VALID + e, 0.0],
        [10.0, S.Y_MAX_VALID - e], [10.0, S.Y_MAX_VALID + e], [10.0, S.Y_MIN_VALID + e], [10.0, S.Y_MIN_VALID - e],
        [S.X_MAX_VALID, S.Y_MIN_VALID], [-8.0, 0.0], [72.0, 0.0], [80.0, 40.0], [float("nan"), 0.0],
        [0.0, float("inf")]], dtype=torch.float64, requires_grad=True)
    v, m = S.sample_sdf(f, pts)
    exp = [True, False, True, False, True, False, True, False, True, False, False, False, False, False]
    assert m.tolist() == exp
    assert (v[~m] == 0).all()
    v.sum().backward()
    assert torch.isfinite(pts.grad).all()
    assert (pts.grad[~m] == 0).all()


def test_sdf_index_and_broadcast():
    fs = torch.from_numpy(_rng(7).normal(0, 3, (2, 320, 256)).astype(np.float32))
    xy = torch.from_numpy(np.stack([_rng(8).uniform(-7, 71, (5, 30)), _rng(9).uniform(-31, 31, (5, 30))], -1)).float()
    idx = torch.tensor([0, 1, 1, 0, 1])
    v, m = S.sample_sdf(fs, xy, idx)
    for b in range(5):
        vb, mb = S.sample_sdf(fs[idx[b]], xy[b])
        assert torch.equal(v[b], vb) and torch.equal(m[b], mb)
    v1, _ = S.sample_sdf(fs[:1], xy)                     # S == 1 broadcasts
    assert torch.equal(v1[3], S.sample_sdf(fs[0], xy[3])[0])
    with pytest.raises(ValueError):
        S.sample_sdf(fs, xy)                             # S=2, B=5 without index


# ----------------------------------------------------------------------------------------- construction
def test_halfplane_analytic_sign_axes_and_gradient():
    import shapely
    # drivable = x < 30 : sdf = 30 - x ;  drivable = y_left > 5 : sdf = y - 5  (checks col 0 = left)
    for geom, fn, grad in ((shapely.box(-100, -100, 30, 100), lambda x, y: 30 - x, (-1.0, 0.0)),
                           (shapely.box(-100, 5, 100, 100), lambda x, y: y - 5, (0.0, 1.0))):
        f = S.rasterize_sdf(geom)
        X, Y = S.grid_mesh()
        assert np.abs(f - np.clip(fn(X, Y), -10, 10)).max() < 1e-5          # float32 storage
        f16 = torch.from_numpy(f.astype(np.float16))
        xy = torch.from_numpy(np.stack([_rng(10).uniform(-7.8, 71.8, 3000), _rng(11).uniform(-31.8, 31.8, 3000)],
                                       -1)).requires_grad_(True)
        v, m = S.sample_sdf(f16, xy)
        ref = np.clip(fn(xy[:, 0].detach().numpy(), xy[:, 1].detach().numpy()), -10, 10)
        lin = np.abs(ref) < 9.7                          # away from the clip kink (bilinear error there)
        assert m.all() and np.abs(v.detach().numpy() - ref)[lin].max() < 5e-3   # float16 rounding only
        v.sum().backward()
        inner = np.abs(ref) < 9.0
        g = xy.grad.numpy()[inner]
        assert np.allclose(g[:, 0], grad[0], atol=1e-2) and np.allclose(g[:, 1], grad[1], atol=1e-2)
    # left/right sanity in plain words: a road on the LEFT gives + at (10, +8) and - at (10, -2)
    f = torch.from_numpy(S.rasterize_sdf(shapely.box(-100, 5, 100, 100)))
    v, _ = S.sample_sdf(f, torch.tensor([[10.0, 8.0], [10.0, -2.0]]))
    assert abs(v[0].item() - 3.0) < 1e-5 and abs(v[1].item() + 7.0) < 1e-5


def test_curved_region_with_hole_vs_exact():
    import shapely
    geom = shapely.Point(30, 0).buffer(20, 64).difference(shapely.box(25, -3, 31, 2))   # disk with a hole
    geom = shapely.union(geom, shapely.box(-20, -4, 12, 4))                              # + a road stub
    f = S.rasterize_sdf(geom)
    X, Y = S.grid_mesh()
    assert np.abs(f - np.clip(S.signed_distance_exact(geom, np.stack([X, Y], -1)), -10, 10)).max() < 1e-5
    xy = np.stack([_rng(12).uniform(-7.8, 71.8, 20000), _rng(13).uniform(-31.8, 31.8, 20000)], -1)
    ex = S.signed_distance_exact(geom, xy)
    v, m = S.sample_sdf_np(f.astype(np.float16), xy)
    near = np.abs(ex) < 2
    err = np.abs(v - ex)[near]
    assert np.median(err) < 0.01 and np.quantile(err, 0.99) < 0.06
    assert (np.sign(v) == np.sign(ex))[np.abs(ex) > 0.05].all()


def test_empty_and_full_regions():
    import shapely
    assert (S.rasterize_sdf(shapely.Polygon()) == -10).all()
    assert (S.rasterize_sdf(shapely.box(-50, -50, 100, 50)) == 10).all()


# ----------------------------------------------------------------------------------------- ego footprint
def test_ego_corners_match_sf_common_and_official():
    sys.path.insert(0, str(ROOT / "report/planner_vs_perception_tests/safety_filter"))
    import sf_common as SF
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import \
        state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    r = _rng(14)
    P = np.stack([r.uniform(-5, 60, 64), r.uniform(-20, 20, 64), r.uniform(-np.pi, np.pi, 64)], -1)
    c = S.ego_corners(torch.from_numpy(P)).numpy()
    assert np.abs(c - SF.ego_corners(P[:, 0], P[:, 1], P[:, 2], 0.0)).max() < 1e-12
    st = np.zeros((1, 64, StateIndex.size()))
    st[0, :, StateIndex.X], st[0, :, StateIndex.Y], st[0, :, StateIndex.HEADING] = P.T
    off = state_array_to_coords_array(st, get_pacifica_parameters())[0, :, :4]   # FL, RL, RR, FR
    assert np.abs(c - off).max() < 1e-9
    v, m = S.corner_sdf(torch.zeros(1, 320, 256), torch.from_numpy(P)[None])
    assert v.shape == (1, 64, 4) and m.shape == (1, 64, 4)


# ----------------------------------------------------------------------------------------- storage
def test_save_load_roundtrip(tmp_path):
    f = _rng(15).normal(0, 3, (320, 256)).astype(np.float32)
    p = S.sdf_path("abc123", "unit", tmp_path)
    n = S.save_sdf(p, f, token="abc123", ego_xyh=(1.0, 2.0, 0.3), info=dict(n_polys=5, n_invalid=0))
    assert n == p.stat().st_size and p.name == "abc123.npz" and p.parent.name == "unit"
    g = S.load_sdf(p)
    assert g.dtype == np.float16 and np.array_equal(g, f.astype(np.float16))
    z = dict(np.load(p))
    z["version"] = np.array("old")
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(ValueError):
        S.load_sdf(tmp_path / "bad.npz")
    assert not list(tmp_path.rglob("*.tmp.npz"))


# ----------------------------------------------------------------------------------------- metric cache
def _navtest_tokens(n_fail=2, n_pass=2):
    import pandas as pd
    t = pd.read_parquet(TABLE, columns=["token", "log", "re_drivable_area_compliance"])
    fail = t[t.re_drivable_area_compliance < 1].iloc[:n_fail]
    ok = t[t.re_drivable_area_compliance == 1].iloc[:n_pass]
    return pd.concat([fail, ok])


needs_mc = pytest.mark.skipif(not (MC_ROOT.is_dir() and TABLE.exists()), reason="navtest metric cache missing")


@needs_mc
def test_metric_cache_field_matches_official_predicate():
    """sign(sdf) at random points == 'inside ANY official-layer polygon' (the scorer's own per-polygon test),
    and the stored float16 field is the exact signed distance at cell centres (+- float16 rounding)."""
    import lzma
    import pickle
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    for _, row in _navtest_tokens(1, 1).iterrows():
        with lzma.open(MC_ROOT / row.log / "unknown" / row.token / "metric_cache.pkl", "rb") as fh:
            mc = pickle.load(fh)
        f, info = S.build_sdf_from_metric_cache(mc)
        assert info["n_polys"] > 0
        dm, xyh = mc.drivable_area_map, info["ego_xyh"]
        idx = dm.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
        xy = np.stack([_rng(16).uniform(-7.8, 71.8, 3000), _rng(17).uniform(-31.8, 31.8, 3000)], -1)
        c, s = np.cos(xyh[2]), np.sin(xyh[2])
        gx, gy = xyh[0] + c * xy[:, 0] - s * xy[:, 1], xyh[1] + s * xy[:, 0] + c * xy[:, 1]
        assert np.abs(S.global_to_n(np.stack([gx, gy], -1), xyh) - xy).max() < 1e-6
        inside_any = np.zeros(len(xy), bool)
        for i in idx:
            inside_any |= shapely.contains_xy(dm._geometries[i], gx, gy)
        v, m = S.sample_sdf_np(f.astype(np.float16), xy)
        union, _ = S.drivable_union(dm, xyh, crop=False)
        ex = S.signed_distance_exact(union, xy)
        clear = np.abs(ex) > 0.05
        assert m.all()
        assert ((v >= 0) == inside_any)[clear].all()
        assert ((ex > 0) == inside_any).mean() > 0.999
        near = np.abs(ex) < 2
        assert np.median(np.abs(v - ex)[near]) < 0.01


@needs_mc
def test_official_scorer_dac_vs_corner_sdf():
    """Official DAC (41 LQR-tracked states) == all tracked corners inside the union; the SDF corner test agrees
    wherever the corners are not within 0.1 m of the boundary."""
    import lzma
    import pickle
    import shapely
    sys.path.insert(0, str(ROOT / "report/collision_counterfactual/counterfactual"))
    import cf_common as CF
    CF.init_worker()
    for _, row in _navtest_tokens(2, 2).iterrows():
        with lzma.open(MC_ROOT / row.log / "unknown" / row.token / "metric_cache.pkl", "rb") as fh:
            mc = pickle.load(fh)
        out, _, _, _ = CF.score(mc, CF.G["traj"][row.token])
        assert out["dac"] == row.re_drivable_area_compliance
        f, info = S.build_sdf_from_metric_cache(mc)
        corners = S.global_to_n(CF.G["scorer"]._ego_coords[1][:, :4], info["ego_xyh"])       # [41, 4, 2]
        union, _ = S.drivable_union(mc.drivable_area_map, info["ego_xyh"], crop=False)
        exact_ok = bool(shapely.contains_xy(union, corners[..., 0], corners[..., 1]).all())
        assert exact_ok == (out["dac"] == 1.0)
        v, m = S.sample_sdf(torch.from_numpy(f.astype(np.float16)), torch.from_numpy(corners))
        vmin = float(v[m].min())
        if abs(vmin) > 0.1:
            assert (vmin >= 0) == (out["dac"] == 1.0)
