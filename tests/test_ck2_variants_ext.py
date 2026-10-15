"""Path extension of the accelerating (mode-B) CK2 variants: variants.make_variants(..., ext=...).

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_variants_ext.py

Checks (synthetic only, CPU, a few seconds):
  * ext='const_curv' (default) is bitwise the decoder's own mode-B decode and bitwise ext omitted;
  * every ext leaves identity / deceleration / lateral-only variants and every mode-B pose with s1 <= S_8 bitwise
    unchanged (and beta / ds4 / ext_m identical);
  * 'straight': poses beyond S_8 lie on the end tangent with the end heading;
  * 'centerline_gt' on a straight route line ('keep' holds the offset, 'decay' merges onto the line) and on a circular
    route (poses beyond S_8 stay on the circle, where 'straight' leaves it); 'centerline_pred' with the circle given
    reversed, split in two pieces (chaining) and next to decoy lines gives the same circle; gate failures fall back
    bitwise to const_curv; the v2 map / metric-cache frame conversions.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import variants as VV  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import N_FREE, decode  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import DraftPath  # noqa: E402

torch.set_num_threads(1)
T8 = torch.arange(1, 9, dtype=torch.float64) * 0.5


def _arc_traj(v, R=None, y0=0.0):
    """Constant-speed candidate along a straight line (R None) or a left circle of radius R through the origin."""
    s = v * T8
    if R is None:
        return torch.stack([s, torch.full_like(s, y0), torch.zeros_like(s)], -1)
    ph = s / R
    return torch.stack([R * torch.sin(ph), R * (1 - torch.cos(ph)), ph], -1)


def _circle_line(R, s_max=150.0, step=0.25, s_min=-10.0):
    s = np.arange(s_min, s_max + 1e-9, step)
    return np.stack([R * np.sin(s / R), R * (1 - np.cos(s / R))], -1)


def _synthetic(T=3, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(T * K):
        v = 2.0 + float(torch.rand(1, generator=g)) * 13.0
        kap = (0.002 + 0.02 * float(torch.rand(1, generator=g))) * (1 if float(torch.rand(1, generator=g)) < .5 else -1)
        s = v * T8
        h = kap * s
        out.append(torch.stack([torch.sin(h) / kap, (1 - torch.cos(h)) / kap, h], -1))
    tau = torch.stack(out).reshape(T, K, 8, 3).float()
    tau[0, 0] = 0.0                                                     # stopped
    return tau, torch.rand(T, generator=g) * 10.0


def _s1_knots(tau, v0, a):
    """Decoder arc length s1 at the 8 knots and S_8 for the mode-B variant a (as make_variants decodes it)."""
    flat = tau.reshape(-1, 8, 3).double()
    N = len(flat)
    S8 = DraftPath(flat).knots()[:, -1]
    o = decode(flat, torch.full((N, N_FREE), float(np.arctanh(a / 2.0)), dtype=torch.float64),
               torch.zeros(N, N_FREE, dtype=torch.float64), v0=v0, mode="B", lat_len=S8)
    return o["s"][:, 5::5], S8, o


def _route_lines(T, line):
    return [[line] for _ in range(T)]


# ----------------------------------------------------------------------------------------------- const_curv
def test_const_curv_default_is_decoder_bitwise():
    tau, v0 = _synthetic()
    kw = dict(speeds=(-0.5, 0.5, 1.0), lats=(-0.5, 0.5), combine="cross")
    a = VV.make_variants(tau, v0, **kw)
    b = VV.make_variants(tau, v0, ext="const_curv", **kw)
    assert torch.equal(a["traj"], b["traj"]) and torch.equal(a["valid"], b["valid"])
    assert a["meta"]["ext"] == "const_curv" and not a["meta"]["ext_on"].any()
    T, K = tau.shape[:2]
    v0r = v0[:, None].expand(T, K).reshape(-1).double()
    for ai, a_ in enumerate((0.5, 1.0)):
        _, _, o = _s1_knots(tau, v0r, a_)
        col = a["meta"]["names"].index(VV.variant_name(a_, 0.0))
        assert torch.equal(a["traj"][:, :, col].reshape(-1, 8, 3), o["traj"].float())
    with pytest.raises(ValueError):
        VV.make_variants(tau, v0, ext="lane")
    with pytest.raises(ValueError):
        VV.make_variants(tau, v0, ext="centerline_gt")                  # needs ext_lines


@pytest.mark.parametrize("ext", ["straight", "centerline_gt", "centerline_pred"])
def test_only_mode_b_beyond_s8_changes(ext):
    tau, v0 = _synthetic()
    T, K = tau.shape[:2]
    kw = dict(speeds=(-1.0, -0.5, 0.5, 1.0), lats=(-0.5, 0.5), combine="cross")
    lines = None
    if ext != "straight":
        lines = [[_circle_line(1 / 0.01)[:, :2] * np.array([1.0, 1.0])] for _ in range(T)]   # some route line
    cc = VV.make_variants(tau, v0, **kw)
    ex = VV.make_variants(tau, v0, ext=ext, ext_lines=lines, **kw)
    tab = VV.variant_table(**kw)
    for k in ("beta", "ds4", "ext_m", "alpha", "d_end"):
        assert torch.equal(cc["meta"][k], ex["meta"][k]), k
    v0r = v0[:, None].expand(T, K).reshape(-1).double()
    n_diff = 0
    for v, (a, d) in enumerate(tab):
        if a <= 0:
            assert torch.equal(cc["traj"][:, :, v], ex["traj"][:, :, v]), VV.variant_name(a, d)
            continue
        s1k, S8, _ = _s1_knots(tau, v0r, a)
        inside = (s1k <= S8[:, None]).reshape(T, K, 8)
        A, B = cc["traj"][:, :, v], ex["traj"][:, :, v]
        assert torch.equal(A[inside], B[inside]), VV.variant_name(a, d)
        n_diff += int((A[~inside] != B[~inside]).any(-1).sum())
    assert n_diff > 0                                                   # the option does something


# ----------------------------------------------------------------------------------------------- straight
def test_straight_geometry():
    tau, v0 = _synthetic(T=4, K=4, seed=3)
    T, K = tau.shape[:2]
    o = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=(), ext="straight")
    bp = DraftPath(tau.reshape(-1, 8, 3).double())
    V8, Te, H8 = bp.V[:, -1], bp.T_end, bp.H[:, -1]
    v0r = v0[:, None].expand(T, K).reshape(-1).double()
    checked = 0
    for ai, a in enumerate((0.5, 1.0)):
        s1k, S8, _ = _s1_knots(tau, v0r, a)
        tr = o["traj"][:, :, 1 + ai].reshape(-1, 8, 3).double()
        for n in range(len(tr)):
            for k in range(8):
                u = float(s1k[n, k] - S8[n])
                if u <= 1e-9 or S8[n] < 2.0:
                    continue
                r = tr[n, k, :2] - V8[n]
                assert abs(float(r[0] * Te[n, 1] - r[1] * Te[n, 0])) < 1e-4                 # on the tangent line
                assert abs(float(r @ Te[n]) - u) < 1e-4                                       # chord = arc
                assert abs(float(tr[n, k, 2] - H8[n])) < 1e-5                                 # end heading
                checked += 1
    assert checked > 10


# ----------------------------------------------------------------------------------------------- centerline_gt
@pytest.mark.parametrize("offset", ["keep", "decay"])
def test_centerline_gt_straight_route_offset_law(offset):
    v, c = 10.0, -0.3                               # candidate along y = 0, route line along y = -0.3 (d0 = +0.3)
    tau = _arc_traj(v)[None, None].float()
    line = np.stack([np.arange(-10.0, 150.0, 0.25), np.full(640, c)], -1)
    o = VV.make_variants(tau, torch.tensor([v]), speeds=(1.0,), lats=(), ext="centerline_gt",
                         ext_lines=[[line]], ext_cfg={"offset": offset})
    assert bool(o["meta"]["ext_on"][0, 0])
    assert abs(float(o["meta"]["ext_d0"][0, 0]) - 0.3) < 1e-6 and abs(float(o["meta"]["ext_psi"][0, 0])) < 1e-9
    s1k, S8, _ = _s1_knots(tau, torch.tensor([v], dtype=torch.float64), 1.0)
    u = float(s1k[0, -1] - S8[0])
    assert u > 2.0
    y8 = float(o["traj"][0, 0, 1, -1, 1])
    if offset == "keep":
        assert abs(y8) < 1e-5                                           # holds the lateral offset (stays in lane)
        assert abs(float(o["traj"][0, 0, 1, -1, 2])) < 1e-5
    else:
        b = 10.0
        exp = c + (0.3 + 2 * 0.3 / b * u) * (1 - u / b) ** 2              # merges onto the line
        assert abs(y8 - exp) < 5e-3 and y8 < -0.05


def test_centerline_gt_follows_circle_straight_does_not():
    R, v = 40.0, 10.0
    tau = _arc_traj(v, R)[None, None].float()
    v0 = torch.tensor([v])
    line = _circle_line(R)
    gt = VV.make_variants(tau, v0, speeds=(1.0,), lats=(), ext="centerline_gt", ext_lines=[[line]])
    st = VV.make_variants(tau, v0, speeds=(1.0,), lats=(), ext="straight")
    s1k, S8, _ = _s1_knots(tau, v0.double(), 1.0)
    beyond = (s1k[0] > S8[0] + 1.0).numpy()
    assert beyond.any()
    ctr = np.array([0.0, R])
    for o, ok in ((gt, True), (st, False)):
        xy = o["traj"][0, 0, 1, :, :2].double().numpy()[beyond]
        err = np.abs(np.hypot(*(xy - ctr).T) - R)
        if ok:
            assert err.max() < 0.02, err
            ph = np.arctan2(xy[:, 0], R - xy[:, 1])                       # heading on the circle at that point
            assert np.abs(o["traj"][0, 0, 1, :, 2].double().numpy()[beyond] - ph).max() < 0.01
        else:
            assert err.max() > 0.1, err


def test_centerline_pred_reversed_split_and_decoys():
    R, v = 40.0, 10.0
    tau = _arc_traj(v, R)[None, None].float()
    v0 = torch.tensor([v])
    line = _circle_line(R, s_max=60.0, s_min=0.0, step=1.6)            # coarse like the v2 head (20 pts ~ 32 m)
    piece1, piece2 = line[:22], line[21:]                             # candidate ends near s = 40 -> needs chaining
    decoy_far = line + np.array([0.0, 6.0])                            # parallel line 6 m to the side (gate)
    decoy_cross = np.stack([np.full(20, 35.0), np.linspace(-20, 20, 20)], -1)   # crosses at 90 deg (gate)
    pred = [decoy_far[::-1], piece2[::-1], decoy_cross, piece1[::-1]]  # undirected: given reversed
    o = VV.make_variants(tau, v0, speeds=(1.0,), lats=(), ext="centerline_pred", ext_lines=[pred])
    assert bool(o["meta"]["ext_on"][0, 0])
    s1k, S8, _ = _s1_knots(tau, v0.double(), 1.0)
    beyond = (s1k[0] > S8[0] + 1.0).numpy()
    xy = o["traj"][0, 0, 1, :, :2].double().numpy()[beyond]
    assert np.abs(np.hypot(*(xy - np.array([0.0, R])).T) - R).max() < 0.05
    assert abs(float(o["meta"]["ext_d0"][0, 0])) < 0.1 and float(o["meta"]["ext_res"][0, 0]) < 0.1


def test_gate_failure_falls_back_bitwise():
    tau, v0 = _synthetic(T=2, K=4, seed=5)
    far = [[np.stack([np.linspace(-50, 50, 50), np.full(50, 30.0)], -1)] for _ in range(2)]   # 30 m to the left
    cc = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=())
    for ext in ("centerline_gt", "centerline_pred"):
        o = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=(), ext=ext, ext_lines=far)
        assert not o["meta"]["ext_on"].any()
        assert torch.equal(o["traj"], cc["traj"])
    o = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=(), ext="centerline_gt", ext_lines=[None, None])
    assert torch.equal(o["traj"], cc["traj"])
    st = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=(), ext="straight")
    o = VV.make_variants(tau, v0, speeds=(0.5, 1.0), lats=(), ext="centerline_gt", ext_lines=far,
                         ext_cfg={"fallback": "straight"})
    assert torch.equal(o["traj"], st["traj"])


# ----------------------------------------------------------------------------------------------- frames
def test_pred_centerlines_frame():
    pts = np.zeros((3, 20, 2))
    xr, yf = np.full(20, 2.0), np.linspace(1.0, 30.0, 20)              # 2 m to the RIGHT, ahead
    pts[1, :, 0] = (xr + 32.0) / 64.0
    pts[1, :, 1] = yf / 32.0
    cls = np.full((3, 4), -5.0)
    cls[1, 2] = 3.0                                                     # centerline query
    cls[2, 0] = 3.0                                                     # road query (other class)
    out = VV.pred_centerlines_local(cls, pts)
    assert len(out) == 1
    assert np.allclose(out[0][:, 0], yf) and np.allclose(out[0][:, 1], -2.0)


def test_route_centerline_frame_and_crop():
    h = 0.7
    ra = SimpleNamespace(x=100.0, y=-50.0, heading=h)
    s = np.arange(-100.0, 300.0, 0.25)
    g = np.stack([100 + s * math.cos(h) - 1.5 * math.sin(h), -50 + s * math.sin(h) + 1.5 * math.cos(h),
                  np.full_like(s, h)], -1)
    mc = SimpleNamespace(centerline=SimpleNamespace(_states_se2_array=g), ego_state=SimpleNamespace(rear_axle=ra))
    loc = VV.route_centerline_local(mc, back=20.0, ahead=160.0)
    assert np.allclose(loc[:, 1], 1.5, atol=1e-9)                       # 1.5 m to the left, parallel
    assert loc[0, 0] >= -20.0 - 1e-6 and loc[-1, 0] <= 160.0 + 1e-6 and loc[-1, 0] > 159.0


def test_pred_chain_rejects_sideways_join_and_curve_never_doubles_back():
    """Regression (real navtrain tokens): a successor starting 1.4 m beside / behind the line end was chained and the
    curve doubled back (pose heading flips of 100-200 deg)."""
    v = 10.0
    tau = _arc_traj(v)[None, None].float()                               # straight along y = 0, ends at x = 40
    first = np.stack([np.linspace(0.0, 40.6, 30), np.zeros(30)], -1)   # runs out 0.6 m after the end
    side = np.stack([np.linspace(39.4, 70.0, 30), np.full(30, 1.4)], -1)    # starts behind, 1.4 m to the left
    o = VV.make_variants(tau, torch.tensor([v]), speeds=(1.0,), lats=(), ext="centerline_pred",
                         ext_lines=[[first, side]])
    assert bool(o["meta"]["ext_on"][0, 0])
    tr = o["traj"][0, 0, 1].double().numpy()
    assert np.abs(tr[:, 2]).max() < 1e-3 and np.abs(tr[:, 1]).max() < 1e-3     # straight continuation, no jump
    # a proper successor (ahead, 0.1 m lateral) is chained and followed
    curve = _circle_line(30.0, s_max=40.0, s_min=0.0, step=0.5) + np.array([40.8, 0.1])
    o2 = VV.make_variants(tau, torch.tensor([v]), speeds=(1.0,), lats=(), ext="centerline_pred",
                          ext_lines=[[first, curve]])
    h2 = o2["traj"][0, 0, 1, -1, 2].item()
    assert 0.02 < h2 < 0.3                                                # turns left with the successor, gently


def test_curve_sanity_gate_rejects_hairpin():
    v = 10.0
    tau = _arc_traj(v)[None, None].float()
    hair = np.concatenate([np.stack([np.linspace(0.0, 41.0, 60), np.zeros(60)], -1),
                           np.stack([np.linspace(41.0, 30.0, 20), np.full(20, 1.0)], -1)])   # U-turn right after
    cc = VV.make_variants(tau, torch.tensor([v]), speeds=(1.0,), lats=())
    o = VV.make_variants(tau, torch.tensor([v]), speeds=(1.0,), lats=(), ext="centerline_gt", ext_lines=[[hair]])
    assert not bool(o["meta"]["ext_on"][0, 0]) and torch.equal(o["traj"], cc["traj"])
