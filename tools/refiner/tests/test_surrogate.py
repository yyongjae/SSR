"""Tests for navsim/agents/para_ssr/refiner/surrogate.py (M7).  CPU, < 1 min.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_surrogate.py

Synthetic tests pin the geometry (box separation vs exact SAT / shapely), masks, weights, gradients (finite, and the
right sign: moving a draft away from an object lowers C_col), the DAC / progress / comfort / modification / gate terms.
Real-data tests (skipped when the stage-T data are missing) use dev tokens: centerline projection vs shapely on the full
metric-cache line, human comfort acceptance (<= 1%) on the dev human table, and a full surrogate_loss backward pass.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report/planner_vs_perception_tests/safety_filter"))
from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as G  # noqa: E402
from navsim.agents.para_ssr.refiner import surrogate as SU  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import dense_reference  # noqa: E402

torch.set_num_threads(1)
DATA = Path("/home/external-user/ssd/yongjae_refiner")
HUMAN_DEV = DATA / "human/dev.npz"
HUMAN_TRAIN = DATA / "human/train.npz"
OBJ_DEV = DATA / "objects/dev"


def _straight(v=8.0, y=0.0, B=1, dtype=torch.float64):
    t = torch.arange(1, 9, dtype=dtype) * 0.5
    tr = torch.stack([v * t, torch.full_like(t, y), torch.zeros_like(t)], -1)
    return tr[None].repeat(B, 1, 1)


def _scene(boxes, obs=None, agent=None, human_overlap=None, dtype=torch.float64):
    """one-scene SceneBatch with static boxes [A, 5] (same at all 41 times)."""
    b = torch.as_tensor(np.asarray(boxes, np.float64), dtype=dtype).reshape(-1, 5)
    A = b.shape[0]
    bx = b[:, None, :].repeat(1, 41, 1)[None]
    obs = torch.ones(1, A, 41, dtype=torch.bool) if obs is None else torch.as_tensor(obs).reshape(1, A, 41)
    agent = torch.ones(1, A, dtype=torch.bool) if agent is None else torch.as_tensor(agent).reshape(1, A)
    hov = None if human_overlap is None else torch.as_tensor(human_overlap).reshape(1, A, 41)
    return SU.SceneBatch(boxes=bx, obs=obs, is_agent=agent, human_overlap=hov)


# ----------------------------------------------------------------------------------------------- separation
def test_box_gaps_vs_sat_and_shapely():
    import shapely
    import sf_common as SF
    rng = np.random.default_rng(0)
    n = 3000
    ego = np.stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(-np.pi, np.pi, n)], -1)
    obj = np.stack([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n), rng.uniform(-4, 4, n), rng.uniform(0.4, 12, n),
                    rng.uniform(0.4, 3.5, n)], -1)
    g, gh = SU.box_separation(torch.as_tensor(ego), torch.as_tensor(obj))
    g, gh = g.numpy(), gh.numpy()
    E = SF.ego_corners(ego[:, 0], ego[:, 1], ego[:, 2], 0.0)
    O = SF.box_corners(obj[:, 0], obj[:, 1], obj[:, 2], obj[:, 3], obj[:, 4])
    ov = SF.sat_overlap(E, O)
    assert np.array_equal(ov, gh <= 0)                       # exact SAT (touching counts)
    pe, po = shapely.polygons(E), shapely.polygons(O)
    dist = shapely.distance(pe, po)
    sep = gh > 0
    assert (gh[sep] <= dist[sep] + 1e-9).all()               # lower bound of the Euclidean gap
    assert np.array_equal(shapely.intersects(pe, po), ~sep)  # same predicate as the official NC geometry
    # smooth, bias corrected: g in [g_hard - T ln4, g_hard]
    assert (g <= gh + 1e-12).all() and (g >= gh - SU.TEMP * math.log(4) - 1e-12).all()


def test_face_separation_and_penetration_exact():
    # object straight ahead, aligned: gap = x_obj - L/2 - (RA2C + HALF_LEN)
    ego = torch.zeros(3)
    for x, L in ((20.0, 4.0), (6.0, 4.0), (5.0, 6.0)):
        b = torch.tensor([x, 0.0, 0.0, L, 2.0], dtype=torch.float64)
        gh = SU.box_separation(ego.double(), b)[1]
        assert abs(float(gh) - (x - L / 2 - SU.RA2C - SU.HALF_LEN)) < 1e-12
    # lateral: object alongside with 0.5 m clearance
    b = torch.tensor([SU.RA2C, 1.1485 + 0.5 + 1.0, 0.0, 4.0, 2.0], dtype=torch.float64)
    assert abs(float(SU.box_separation(ego.double(), b)[1]) - 0.5) < 1e-12


# ----------------------------------------------------------------------------------------------- collision
def test_collision_masks_and_weights():
    tr = _straight(8.0)                                      # 32 m in 4 s
    dense = dense_reference(tr)
    car = [16.0, 0.0, 0.0, 4.5, 2.0]                         # hit around t = 1.5-2 s
    base = SU.collision_cost(dense, _scene([car]))
    assert base["cost"].item() > 0 and base["viol"].item() > 0 and bool(base["hard_overlap"].item())
    assert base["first_n"].item() > 0 and base["first_obj"].item() == 0
    # static weight = 0.5 x agent
    st = SU.collision_cost(dense, _scene([car], agent=[False]))
    assert abs(st["cost"].item() - 0.5 * base["cost"].item()) < 1e-12
    # ABSENT (not OBS) -> nothing
    absent = SU.collision_cost(dense, _scene([car], obs=np.zeros((1, 41), bool)))
    assert absent["cost"].item() == 0 and absent["viol"].item() == -math.inf
    # human overlap mask removes exactly the masked pairs
    hov = np.zeros((1, 41), bool)
    hov[0, :] = True
    hm = SU.collision_cost(dense, _scene([car], human_overlap=hov))
    assert hm["cost"].item() == 0 and hm["n_human_masked"].item() > 0
    off = SU.collision_cost(dense, _scene([car], human_overlap=hov), cfg=SU.SurrogateConfig(use_human_mask=False))
    assert abs(off["cost"].item() - base["cost"].item()) < 1e-12
    # behind: object 6 m behind the ego centre, moving with it -> never counted
    bx = torch.as_tensor(dense[0, :, :2].numpy() + np.array([SU.RA2C - 6.0, 0.0]))
    boxes = torch.cat([bx, torch.zeros(41, 1, dtype=torch.float64), torch.tensor([[4.5, 2.0]]).double().repeat(41, 1)],
                      -1)[None, None]
    sc = SU.SceneBatch(boxes=boxes, obs=torch.ones(1, 1, 41, dtype=torch.bool), is_agent=torch.ones(1, 1, dtype=torch.bool))
    assert SU.collision_cost(dense, sc)["cost"].item() == 0
    assert SU.collision_cost(dense, sc, cfg=SU.SurrogateConfig(behind_m=None))["cost"].item() > 0
    # no objects at all
    empty = SU.SceneBatch(boxes=torch.zeros(1, 0, 41, 5, dtype=torch.float64), obs=torch.zeros(1, 0, 41, dtype=torch.bool),
                          is_agent=torch.zeros(1, 0, dtype=torch.bool))
    e = SU.collision_cost(dense, empty)
    assert e["cost"].item() == 0 and e["viol"].item() == -math.inf


def test_collision_far_is_zero_and_margin_flag():
    tr = _straight(8.0)
    dense = dense_reference(tr)
    far = SU.collision_cost(dense, _scene([[16.0, 30.0, 0.0, 4.5, 2.0]]))
    assert far["cost"].item() < 1e-30 and far["viol"].item() < 0
    # object alongside with 0.2 m clearance at every time -> flagged at m_col = 0.3, not a hard overlap
    y = SU.HALF_WID + 0.2 + 1.0
    boxes = dense[0, :, :2].numpy() + np.array([SU.RA2C, y])
    b = torch.as_tensor(np.concatenate([boxes, np.zeros((41, 1)), np.tile([[4.0, 2.0]], (41, 1))], 1))[None, None]
    sc = SU.SceneBatch(boxes=b, obs=torch.ones(1, 1, 41, dtype=torch.bool), is_agent=torch.ones(1, 1, dtype=torch.bool))
    o = SU.collision_cost(dense, sc)
    assert o["viol"].item() > 0 and not bool(o["hard_overlap"].item())
    assert abs(o["gmin_hard"].item() - 0.2) < 1e-9


def test_collision_gradient_moves_away():
    """Moving a draft away from an object lowers C_col, and -grad points away from it."""
    tr = _straight(8.0, y=0.0)
    for car, away in (([16.0, 1.2, 0.0, 4.5, 2.0], torch.tensor([0.0, -1.0]).double()),      # object left-ahead -> go right
                      ([16.0, -1.2, 0.0, 4.5, 2.0], torch.tensor([0.0, 1.0]).double())):     # right-ahead -> go left
        sc = _scene([car])
        dense = dense_reference(tr).requires_grad_(True)
        c = SU.collision_cost(dense, sc)["cost"].sum()
        c.backward()
        g = dense.grad[0, :, :2]
        assert torch.isfinite(dense.grad).all()
        act = g.norm(dim=-1) > 1e-6
        assert act.any()
        assert ((-g[act]) @ away > 0).all()                              # descent direction points away
        moved = dense_reference(tr + torch.cat([away, torch.zeros(1)]).double() * 0.5)
        assert SU.collision_cost(moved, sc)["cost"].item() < c.item()
        stepped = (dense - 0.2 * dense.grad / dense.grad.norm()).detach()
        assert SU.collision_cost(stepped, sc)["cost"].item() < c.item()


def test_collision_gradient_through_decoder():
    """Object ahead in lane: braking (z < 0 direction of -grad) lowers C_col; lateral object: w moves away."""
    tr = _straight(8.0).float()
    # lon: the ego only reaches the car's rear near the end (approach phase only -> every Q_i pushes into it)
    for car, check in (([34.0, 0.0, 0.0, 4.5, 2.0], "lon"), ([18.0, 1.6, 0.0, 4.5, 2.0], "lat")):
        sc = _scene([car], dtype=torch.float32)
        z = torch.zeros(1, 6, requires_grad=True)
        w = torch.zeros(1, 6, requires_grad=True)
        dec = D.decode(tr, z, w, torch.tensor([8.0]), "A")
        c = SU.collision_cost(dec["dense"], sc)["cost"].sum()
        c.backward()
        assert torch.isfinite(z.grad).all() and torch.isfinite(w.grad).all()
        if check == "lon":
            assert (z.grad >= -1e-9).all() and z.grad.abs().sum() > 0     # -grad = brake
        else:
            assert w.grad.abs().sum() > 0
        with torch.no_grad():
            d2 = D.decode(tr, -0.3 * z.grad / (z.grad.norm() + 1e-12), -0.3 * w.grad / (w.grad.norm() + 1e-12),
                          torch.tensor([8.0]), "A")
            assert SU.collision_cost(d2["dense"], sc)["cost"].item() < c.item()


def test_collision_gradcheck_and_nonfinite_safety():
    tr = _straight(8.0, y=0.3)
    sc = _scene([[16.0, 1.5, 0.3, 4.5, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]], obs=np.stack([np.ones(41, bool), np.zeros(41, bool)]))
    dense = dense_reference(tr).requires_grad_(True)
    cfg = SU.SurrogateConfig(behind_m=None)                  # the boolean mask is not differentiable anyway
    assert torch.autograd.gradcheck(lambda d: SU.collision_cost(d, sc, cfg=cfg, details=False)["cost"], (dense,),
                                    eps=1e-6, atol=1e-6)
    # exactly aligned centres (|.| kink) and a zero-size padded box: finite gradients
    sc2 = _scene([[8.0, 0.0, 0.0, 4.5, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    d2 = dense_reference(_straight(8.0)).requires_grad_(True)
    SU.collision_cost(d2, sc2)["cost"].sum().backward()
    assert torch.isfinite(d2.grad).all()


def test_human_overlap_mask_matches_sat():
    import sf_common as SF
    rng = np.random.default_rng(3)
    A = 40
    boxes = np.stack([rng.uniform(-5, 40, A), rng.uniform(-4, 4, A), rng.uniform(-3, 3, A), rng.uniform(1, 6, A),
                      rng.uniform(0.5, 2.5, A)], -1)
    hd = dense_reference(_straight(8.0, y=0.2))
    b = torch.as_tensor(boxes)[:, None, :].repeat(1, 41, 1)[None]
    m = SU.human_overlap_mask(hd, b)[0].numpy()
    x, y, h = (hd[0, :, i].numpy() for i in range(3))
    E = SF.ego_corners(x, y, h, 0.0)
    for j in range(A):
        O = SF.box_corners(np.full(41, boxes[j, 0]), np.full(41, boxes[j, 1]), np.full(41, boxes[j, 2]), boxes[j, 3],
                           boxes[j, 4])
        assert np.array_equal(m[j], SF.sat_overlap(E, O))


def test_scene_index_sharing():
    """13 drafts sharing one scene == 13 scenes."""
    tr = _straight(8.0, B=4) + torch.tensor([0.0, 0.3, 0.0]).double() * torch.arange(4).double()[:, None, None]
    sc = _scene([[16.0, 1.0, 0.0, 4.5, 2.0]])
    a = SU.collision_cost(dense_reference(tr), sc, index=torch.zeros(4, dtype=torch.long))["cost"]
    rep = SU.SceneBatch(boxes=sc.boxes.repeat(4, 1, 1, 1), obs=sc.obs.repeat(4, 1, 1), is_agent=sc.is_agent.repeat(4, 1))
    b = SU.collision_cost(dense_reference(tr), rep)["cost"]
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------------------------- DAC
def _road_sdf(half=3.0):
    """E-grid field of a straight road |y| < half (exact signed distance)."""
    from navsim.agents.para_ssr.refiner import sdf as S
    X, Y = S.grid_mesh()
    return torch.as_tensor((half - np.abs(Y)).astype(np.float32))


def test_dac_cost_sign_gradient_and_oog():
    sdf = _road_sdf(3.0)[None]
    sc = SU.SceneBatch(sdf=sdf)
    inside = SU.dac_cost(dense_reference(_straight(8.0)), sc)
    assert inside["sdf_min"].item() > 1.8 - 1e-3 and inside["viol"].item() < 0 and inside["n_oog"].item() == 0
    shifted = dense_reference(_straight(8.0, y=2.2)).requires_grad_(True)
    o = SU.dac_cost(shifted, sc)
    assert o["viol"].item() > 0 and bool(o["hard_out"].item())
    o["cost"].sum().backward()
    assert torch.isfinite(shifted.grad).all() and (shifted.grad[0, 1:, 1] > 0).all()   # -grad pushes back to y = 0
    fast = dense_reference(_straight(22.0))                                             # x up to 88 m: off the grid
    f = SU.dac_cost(fast, sc)
    assert f["n_oog"].item() > 0 and torch.isfinite(f["cost"]).all()


# ----------------------------------------------------------------------------------------------- progress
def test_projection_matches_shapely():
    from shapely.geometry import LineString, Point
    rng = np.random.default_rng(5)
    for _ in range(20):
        L = rng.integers(3, 30)
        ang = np.cumsum(rng.normal(0, 0.3, L))
        step = rng.uniform(0.0, 3.0, L)
        step[rng.random(L) < 0.1] = 0.0                              # duplicate vertices like the metric cache
        pts = np.cumsum(np.stack([step * np.cos(ang), step * np.sin(ang)], -1), 0)
        q = rng.uniform(pts.min(0) - 3, pts.max(0) + 3, (50, 2))
        s, _ = SU.project_on_polyline(torch.as_tensor(q)[None], torch.as_tensor(pts)[None])
        ls = LineString(pts)
        ref = np.array([ls.project(Point(*p)) for p in q])
        assert np.abs(s[0].numpy() - ref).max() < 1e-9
    # padding ignored
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    pad = np.concatenate([pts, np.array([[5.0, 1.0], [5.0, 2.0]])])
    valid = torch.tensor([[True, True, True, False, False]])
    s, _ = SU.project_on_polyline(torch.tensor([[[5.0, 1.5]]]).double(), torch.as_tensor(pad)[None], valid)
    assert abs(s.item() - 5.0) < 1e-12


def test_progress_and_ep_rules():
    cl = torch.stack([torch.linspace(-30, 150, 181), torch.zeros(181)], -1).double()[None]
    sc = SU.SceneBatch(centerline=cl, cl_valid=torch.ones(1, 181, dtype=torch.bool))
    for v in (0.0, 4.0, 9.0):
        P = SU.progress(dense_reference(_straight(v)), sc)
        assert abs(P.item() - v * 4.0) < 1e-9
    P = torch.tensor([0.0, 3.0, 20.0, 40.0, 30.0]).double()
    pp = torch.tensor([0.0, 4.0, 30.0, 30.0, 0.0]).double()
    ep = SU.ep_surrogate(P, pp)
    assert torch.allclose(ep, torch.tensor([1.0, 1.0, 20 / 30, 1.0, 1.0]).double())
    # mode A loss: slowing to P_pdm is free, below it costs the EP drop
    P0 = torch.tensor([40.0, 40.0, 10.0]).double()
    P1 = torch.tensor([30.0, 20.0, 5.0]).double()
    pp = torch.tensor([30.0, 30.0, 2.0]).double()
    L = SU.progress_loss(P1, P0, pp)
    assert torch.allclose(L, torch.tensor([0.0, 10 / 30, 0.0]).double())
    assert SU.progress_loss(P1, P0, pp, drop=torch.tensor([False, True, False]))[1].item() == 0


# ----------------------------------------------------------------------------------------------- comfort / mod
def test_keyframe_comfort_matches_decoder_kinematics():
    rng = np.random.default_rng(2)
    for _ in range(50):
        v = rng.uniform(0, 15)
        acc = rng.normal(0, 1.5, 8)
        u = np.clip(v + np.cumsum(acc) * 0.5, 0.1, None)
        h = np.cumsum(rng.normal(0, 0.05, 8))
        xy = np.cumsum(np.stack([u * 0.5 * np.cos(h), u * 0.5 * np.sin(h)], -1), 0)
        tr = np.concatenate([xy, h[:, None]], 1)
        k = D.keyframe_kinematics(tr, v)
        kc = SU.keyframe_comfort(torch.as_tensor(tr)[None], torch.tensor([v], dtype=torch.float64))
        assert abs(kc["lon_acc"].max().item() - k["acc_max"]) < 1e-9 and abs(kc["lon_acc"].min().item() - k["acc_min"]) < 1e-9
        assert abs(kc["lat_acc"].abs().max().item() - k["lat_max"]) < 1e-9
        assert abs(kc["yaw_rate"].abs().max().item() - k["yaw_max"]) < 1e-9


def test_comfort_penalty_thresholds():
    x = {"lon_acc": torch.tensor([[2.4, -4.05]], dtype=torch.float64), "lat_acc": torch.tensor([[4.0]], dtype=torch.float64)}
    p = SU.comfort_penalty(x, 1.0)
    assert p["cost"].item() == 0 and not bool(p["flag"].item())
    p = SU.comfort_penalty(x, 0.9)
    exp = ((2.4 - 0.9 * 2.4) ** 2 / 2.4 ** 2 + (4.05 - 0.9 * 4.05) ** 2 / 4.05 ** 2) / 2 + (4.0 - 0.9 * 4.89) ** 2 / 4.89 ** 2 * 0
    assert abs(p["cost"].item() - exp) < 1e-12 and bool(p["flag"].item())


def test_modification_and_identity():
    tr = _straight(8.0, B=2).float()
    z = torch.zeros(2, 6)
    w = torch.zeros(2, 6)
    dec = D.decode(tr, z, w, torch.tensor([8.0, 8.0]), "A")
    assert SU.modification_cost(dec).abs().max().item() == 0
    an = SU.comfort_penalty(SU.analytic_comfort(dec), SU.CMF_FRAC_AN)
    assert an["cost"].abs().max().item() == 0
    dec2 = D.decode(tr, torch.full((2, 6), -1.0), torch.full((2, 6), 0.3), torch.tensor([8.0, 8.0]), "A")
    assert (SU.modification_cost(dec2) > 0).all()


# ----------------------------------------------------------------------------------------------- gate
def test_gate_labels_weight_and_detach():
    y = SU.gate_labels(np.array([1, 0.5, 1, 1]), np.array([1, 1, 0, 1]), np.array([1, 1, 1, 0.5]))
    assert y.tolist() == [0.0, 1.0, 1.0, 1.0]
    y2 = SU.gate_labels(np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), ttc=np.array([0, 1]), use_ttc=True)
    assert y2.tolist() == [1.0, 0.0]
    assert SU.gate_pos_weight(0.25) == 3.0 and SU.gate_pos_weight(0.01) == 10.0
    trunk = torch.nn.Linear(8, 16)
    head = SU.GateHead(16, 32)
    x = torch.randn(5, 8)
    feat = trunk(x)
    logit = head(feat)
    yy = torch.tensor([1.0, 0, 0, 1, 0])
    loss = SU.gate_loss(logit, yy, prior=0.2)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logit, yy, pos_weight=torch.tensor(4.0))
    assert abs(loss.item() - ref.item()) < 1e-6
    loss.backward()
    assert trunk.weight.grad is None and head.mlp[0].weight.grad is not None


# ----------------------------------------------------------------------------------------------- UNKNOWN
def test_unknown_footprint_matches_numpy():
    from navsim.agents.para_ssr.refiner.sdf import ego_corners
    rng = np.random.default_rng(4)
    obj = {"ego_kf": np.cumsum(rng.normal([4, 0, 0], [1, .3, .02], (11, 3)), 0).astype(np.float32),
           "n_kf": np.int32(11), "R": np.float32(85.0)}
    fac = rng.uniform(0.3, 6.0, (64, 1, 1))                 # up to ~36 m/s: some drafts leave the 75 m reach
    trajs = np.cumsum(rng.normal([3, 0, 0], [0.5, 1.0, 0.05], (64, 8, 3)), 1) * fac
    dense = dense_reference(torch.as_tensor(trajs))
    t = np.arange(41) * 0.1
    cor = ego_corners(dense[:, 1:]).numpy()                  # [B, 40, 4, 2]
    for n_kf in (11, 7):                                     # 7 keyframes -> t_avail = 3 s: everything later UNKNOWN
        obj["n_kf"] = np.int32(n_kf)
        sc = SU.SceneBatch(gt_ego=torch.as_tensor(G.gt_ego_xy(obj, t))[None],
                           t_avail=torch.tensor([G.t_avail(obj)]).double(), R=torch.tensor([85.0]).double())
        for rm in (10.0, None):
            got = SU.unknown_footprint(dense, sc, torch.zeros(64, dtype=torch.long), SU.SurrogateConfig(radius_margin=rm))
            ref = G.unknown_space(obj, cor.transpose(0, 2, 1, 3), t[1:], radius_margin=rm).any(-1).any(-1)
            assert np.array_equal(got.numpy(), ref)
            if n_kf == 11:
                assert ref.any() and (~ref).any()
            else:
                assert ref.all()


# ----------------------------------------------------------------------------------------------- real data
def _human(path):
    z = np.load(path)
    ok = ~z["frame_gap"]
    return z["traj"][ok], z["v0"][ok], z["a0"][ok]


@pytest.mark.skipif(not HUMAN_DEV.exists(), reason="dev human table missing")
def test_comfort_human_acceptance():
    """IMPL_SPEC §3.7 acceptance: human trajectories violate the comfort surrogate in <= 1% (keyframe + analytic
    terms at identity).  Dev and train human tables (31,750 tokens)."""
    for path in (HUMAN_DEV, HUMAN_TRAIN):
        if not path.exists():
            continue
        tr, v0, a0 = _human(path)
        flags = []
        for s in range(0, len(tr), 2048):
            t = torch.as_tensor(tr[s:s + 2048])
            vv = torch.as_tensor(v0[s:s + 2048])
            dec = D.decode(t, torch.zeros(len(t), 6), torch.zeros(len(t), 6), vv, "A")
            c = SU.comfort_cost(t, vv, torch.as_tensor(a0[s:s + 2048]), dec)
            flags.append((c["kf_flag"] | c["an_flag"]).numpy())
        rate = np.concatenate(flags).mean()
        assert rate <= 0.01, (path.name, rate)


@pytest.mark.skipif(not OBJ_DEV.exists(), reason="dev objects missing")
def test_real_token_end_to_end():
    import pandas as pd
    sys.path.insert(0, str(ROOT / "tools/refiner"))
    import score_trajectories as ST
    from shapely.geometry import Point
    from navsim.agents.para_ssr.refiner import sdf as S
    toks = sorted(p.stem for p in OBJ_DEV.glob("*.npz"))[:2]
    if not toks:
        pytest.skip("no dev objects")
    hz = np.load(HUMAN_DEV)
    hidx = {t: i for i, t in enumerate(hz["tokens"].tolist())}
    split = pd.read_parquet(DATA / "splits/dev.parquet").set_index("token")
    for tok in toks:
        mc = ST.load_metric_cache(ST.locate_metric_cache(tok, split.loc[tok, "log"]))
        objs = G.load_objects(OBJ_DEV / f"{tok}.npz")
        sdfp = S.sdf_path(tok, "navtrain")
        if not sdfp.exists():
            continue
        cl = SU.centerline_from_metric_cache(mc)
        i = hidx[tok]
        th = hz["traj"][i]
        scene = SU.collate_scenes([SU.scene_from_numpy(objs, S.load_sdf(sdfp), cl, th, 30.0, hz["v0"][i], hz["a0"][i])])
        bank = D.sample_bank(th, tok, path_long=hz["path"][i], n_valid=int(hz["n_reg"][i]), v0=float(hz["v0"][i]),
                             a0=float(hz["a0"][i]))
        drafts = torch.as_tensor(bank["drafts"]).double()
        idx = torch.zeros(len(drafts), dtype=torch.long)
        # centerline projection (cropped, torch) == shapely on the FULL metric-cache line (global frame)
        ev = SU.evaluate_trajectories(drafts, scene, idx)
        ra = mc.ego_state.rear_axle
        c, s = math.cos(ra.heading), math.sin(ra.heading)
        cen = SU.ego_centre(ev["dense"][:, [0, -1]]).numpy()
        gx, gy = ra.x + c * cen[..., 0] - s * cen[..., 1], ra.y + s * cen[..., 0] + c * cen[..., 1]
        ref = np.array([[mc.centerline.project([Point(gx[b, j], gy[b, j])])[0] for j in range(2)] for b in range(len(cen))])
        assert np.abs(ev["P"].numpy() - np.clip(ref[:, 1] - ref[:, 0], 0, None)).max() < 1e-6
        # the human (identity draft) never collides with an object it overlaps itself (mask) -> hard overlap False
        assert not bool(ev["col"]["hard_overlap"][0])
        # full loss backward through the decoder: finite
        z = torch.zeros(len(drafts), 6, requires_grad=True)
        w = torch.zeros(len(drafts), 6, requires_grad=True)
        dec = D.decode(drafts, z, w, torch.full((len(drafts),), float(hz["v0"][i])).double(), "A")
        tot, terms = SU.surrogate_loss(dec, drafts, scene, idx)
        tot.backward()
        assert torch.isfinite(tot) and torch.isfinite(z.grad).all() and torch.isfinite(w.grad).all()
