"""Tests for navsim/agents/para_ssr/refiner/gt_future.py (CPU, < 1 min).

Synthetic logs: objects are defined in the GLOBAL frame, written into each frame's ego (rear-axle) frame like the raw
NAVSIM logs, and the builder must recover them in the t0 N frame (centre, heading, velocity rotation, first-appearance
size, presence rule, singleton, gaps, radius).  One real navtest token is checked against its metric cache in
test_build_future_objects.py.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report/planner_vs_perception_tests/safety_filter"))

from navsim.agents.para_ssr.refiner import gt_future as G  # noqa: E402


# ------------------------------------------------------------------------------------------------ synthetic log
def _quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _to_local(gx, gy, gh, gvx, gvy, pose):
    x, y, h = pose
    c, s = math.cos(h), math.sin(h)
    dx, dy = gx - x, gy - y
    return (c * dx + s * dy, -s * dx + c * dy, gh - h, c * gvx + s * gvy, -s * gvx + c * gvy)


def ego_pose(k, speed=4.0, x0=1000.0, y0=2000.0, h0=3.0, yaw_rate=0.05):
    """ego global pose at frame k (0.5 s steps): arc of constant speed / yaw rate."""
    t = 0.5 * k
    h = h0 + yaw_rate * t
    # integrate exactly for a circular arc
    x = x0 + speed / yaw_rate * (math.sin(h) - math.sin(h0))
    y = y0 - speed / yaw_rate * (math.cos(h) - math.cos(h0))
    return (x, y, h)


def obj_global(name, k, pose0):
    """global state (x, y, heading, vx, vy, L, W) of synthetic object ``name`` at frame k, or None if absent."""
    x0, y0, h0 = pose0
    c, s = math.cos(h0), math.sin(h0)

    def at(xn, yn):  # N -> global
        return x0 + c * xn - s * yn, y0 + s * xn + c * yn

    t = 0.5 * k
    if name == "veh_a":          # present 0..10, heading crosses +-pi in N, moving
        hn = math.pi - 0.3 + 0.1 * k
        vn = (3.0 * math.cos(hn), 3.0 * math.sin(hn))
        gx, gy = at(20.0 + vn[0] * t, 5.0 + vn[1] * t)
        return (gx, gy, hn + h0, c * vn[0] - s * vn[1], s * vn[0] + c * vn[1], 4.5 + 0.01 * k, 1.9 + 0.02 * k)
    if name == "ped_b" and 3 <= k <= 7:
        gx, gy = at(-10.0, -3.0 + 1.2 * (t - 1.5))
        return (gx, gy, h0 + math.pi / 2, -s * 1.2, c * 1.2, 0.6, 0.7)
    if name == "cone_c" and k == 4:   # singleton static with (spurious) log velocity
        gx, gy = at(15.0, -2.0)
        return (gx, gy, h0 + 0.2, 0.3, -0.4, 0.4, 0.4)
    if name == "far_d":          # always beyond R (from the origin)
        gx, gy = at(-150.0, 30.0)
        return (gx, gy, h0, 0.0, 0.0, 4.0, 2.0)
    if name == "late_e" and k >= 2:   # outside R at k=2..5, enters R at k >= 6
        gx, gy = at(120.0 - 12.0 * (k - 2), 0.0)
        return (gx, gy, h0 + math.pi, -12.0 * c, -12.0 * s, 5.0, 2.0)
    if name == "gap_f" and k in (4, 6):   # annotated at 4 and 6, missing at 5
        gx, gy = at(30.0 + 2.0 * (k - 4), -8.0)
        return (gx, gy, h0 + 0.1 * k, 2.0 * c, 2.0 * s, 4.0, 1.8)
    return None


NAMES = {"veh_a": "vehicle", "ped_b": "pedestrian", "cone_c": "traffic_cone", "far_d": "vehicle",
         "late_e": "bicycle", "gap_f": "vehicle"}


def make_log(n_frames=20, speed=4.0, dup_at=None):
    pose0 = ego_pose(0, speed)
    frames = []
    for k in range(n_frames):
        pose = ego_pose(k, speed)
        boxes, names, vel, toks = [], [], [], []
        for nm in NAMES:
            st = obj_global(nm, k, pose0)
            if st is None:
                continue
            lx, ly, lh, lvx, lvy = _to_local(st[0], st[1], st[2], st[3], st[4], pose)
            boxes.append([lx, ly, 0.5, st[5], st[6], 1.5, lh])
            names.append(NAMES[nm])
            vel.append([lvx, lvy, 0.0])
            toks.append(nm)
        if dup_at is not None and k == dup_at:        # duplicate track token (2nd copy must be ignored)
            boxes.append([boxes[0][0] + 50, boxes[0][1], 0.5, 1, 1, 1, 0])
            names.append("vehicle"); vel.append([0, 0, 0]); toks.append(toks[0])
        frames.append(dict(token=f"tok{k:03d}", timestamp=int(1e6 * 0.5 * k), frame_idx=k,
                           ego2global_translation=[pose[0], pose[1], 0.0], ego2global_rotation=_quat(pose[2]),
                           ego_dynamic_state=[speed, 0.0, 0.0, 0.0],
                           anns=dict(gt_boxes=np.array(boxes, np.float64).reshape(-1, 7), gt_names=np.array(names),
                                     gt_velocity_3d=np.array(vel, np.float64).reshape(-1, 3),
                                     track_tokens=toks, instance_tokens=[f"i{k}_{j}" for j in range(len(toks))])))
    return frames, pose0


def expected_n(nm, k, pose0):
    st = obj_global(nm, k, pose0)
    return None if st is None else _to_local(st[0], st[1], st[2], st[3], st[4], pose0)


@pytest.fixture(scope="module")
def built():
    frames, pose0 = make_log()
    return G.build_from_frames(frames[:17]), pose0


# ------------------------------------------------------------------------------------------------ tests
def test_constants_match_nuplan():
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
    from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
    for name, cid in G.CLASS_ID.items():
        assert tracked_object_types[name].value == cid
    assert set(G.AGENT_CLASSES) == {t.value for t in AGENT_TYPES}
    assert TrackedObjectType.EGO.value == G.CLASS_ID["ego"]


def test_build_structure_and_radius(built):
    obj, _ = built
    tr = list(obj["track"])
    assert "far_d" not in tr, "track beyond R at every keyframe must be dropped"
    assert set(tr) == {"veh_a", "ped_b", "cone_c", "late_e", "gap_f"}
    A = len(tr)
    assert obj["kf"].shape == (A, 11, 6) and obj["kf"].dtype == np.float32
    assert obj["first"].shape == (A, 6) and obj["first"].dtype == np.float32
    assert obj["meta"].shape == (A, 5) and obj["meta"].dtype == np.int16
    assert obj["ego_kf"].shape == (11, 3)
    assert int(obj["n_kf"]) == 11 and int(obj["n_tracks_all"]) == 6 and int(obj["n_dup"]) == 0
    # S_avail = arc length over 16 frames (8 s) at 4 m/s = 32 m (chord sum of a gentle arc) -> R = 80
    assert abs(float(obj["S_avail"]) - 32.0) < 1e-2
    assert float(obj["R"]) == 80.0
    # ordering: ascending distance to the GT ego path (keyframe positions)
    e = obj["ego_kf"][:, :2].astype(np.float64)
    d = []
    for i in range(A):
        k = obj["kf"][i, :, G.KF_PRESENT] > 0
        d.append(np.linalg.norm(obj["kf"][i, k, None, :2] - e[None], axis=-1).min())
    assert (np.diff(d) >= -1e-4).all()


def test_radius_grows_with_path():
    frames, _ = make_log(n_frames=20, speed=12.0)
    obj = G.build_from_frames(frames[:17])
    assert abs(float(obj["S_avail"]) - 96.0) < 0.2
    assert abs(float(obj["R"]) - (float(obj["S_avail"]) + 25.0)) < 1e-4


def test_centres_headings_velocities(built):
    obj, pose0 = built
    tr = list(obj["track"])
    for nm in tr:
        i = tr.index(nm)
        ks = [k for k in range(11) if expected_n(nm, k, pose0) is not None]
        for k in ks:
            ex = expected_n(nm, k, pose0)
            kf = obj["kf"][i, k]
            assert abs(kf[G.KF_X] - ex[0]) < 1e-4 and abs(kf[G.KF_Y] - ex[1]) < 1e-4
            assert abs(G.wrap(kf[G.KF_H] - ex[2])) < 1e-5
            agent = obj["meta"][i, G.MT_AGENT] == 1
            evx, evy = (ex[3], ex[4]) if agent else (0.0, 0.0)
            assert abs(kf[G.KF_VX] - evx) < 1e-4 and abs(kf[G.KF_VY] - evy) < 1e-4, nm
            assert kf[G.KF_PRESENT] == 1.0
        # heading unwrapped along the track
        h = obj["kf"][i, ks[0]:ks[-1] + 1, G.KF_H]
        assert np.all(np.abs(np.diff(h)) < math.pi)


def test_heading_unwrap_crosses_pi(built):
    obj, _ = built
    i = list(obj["track"]).index("veh_a")
    h = obj["kf"][i, :, G.KF_H]
    assert h.max() > math.pi, "veh_a crosses +pi: unwrapped heading must exceed pi"
    assert np.allclose(np.diff(h), 0.1, atol=1e-5)


def test_first_appearance_and_meta(built):
    obj, pose0 = built
    tr = list(obj["track"])
    i = tr.index("veh_a")
    assert abs(obj["first"][i, G.FI_L] - 4.5) < 1e-6 and abs(obj["first"][i, G.FI_W] - 1.9) < 1e-6
    i = tr.index("ped_b")
    assert list(obj["meta"][i]) == [G.CLASS_ID["pedestrian"], 1, 3, 7, 0]
    ex = expected_n("ped_b", 3, pose0)
    assert abs(obj["first"][i, G.FI_H] - G.wrap(ex[2])) < 1e-5
    assert abs(obj["first"][i, G.FI_VY] - ex[4]) < 1e-4 and obj["first"][i, G.FI_K] == 3
    i = tr.index("cone_c")
    assert list(obj["meta"][i]) == [G.CLASS_ID["traffic_cone"], 0, 4, 4, 1]
    assert obj["first"][i, G.FI_VX] == 0 and obj["first"][i, G.FI_VY] == 0, "static objects: velocity 0"
    i = tr.index("late_e")
    assert list(obj["meta"][i]) == [G.CLASS_ID["bicycle"], 1, 2, 10, 0]


def test_gap_filled_linearly(built):
    obj, pose0 = built
    i = list(obj["track"]).index("gap_f")
    kf = obj["kf"][i]
    assert kf[5, G.KF_PRESENT] == 0 and kf[4, G.KF_PRESENT] == 1 and kf[6, G.KF_PRESENT] == 1
    assert np.allclose(kf[5, :5], 0.5 * (kf[4, :5] + kf[6, :5]), atol=1e-5)
    b, st = G.query(obj, np.array([2.5]))
    assert st[i, 0] == G.OBS


def test_query_rule(built):
    obj, pose0 = built
    tr = list(obj["track"])
    t51 = np.arange(51) * 0.1
    boxes, state, unk = G.query(obj, t51, dtype=np.float64, return_unknown=True)
    assert boxes.shape == (len(tr), 51, 5) and state.dtype == np.int8 and not unk.any()
    # knots reproduce keyframes exactly (float64 of stored f32)
    for i in range(len(tr)):
        fk, lk = obj["meta"][i, G.MT_FIRST], obj["meta"][i, G.MT_LAST]
        for k in range(fk, lk + 1):
            assert np.array_equal(boxes[i, 5 * k, :3], obj["kf"][i, k, :3].astype(np.float64))
            assert np.array_equal(boxes[i, 5 * k, 3:], obj["first"][i, :2].astype(np.float64))
    # multi-keyframe presence = [5 fk, 5 lk]
    i = tr.index("ped_b")
    exp = np.zeros(51, bool); exp[15:36] = True
    assert np.array_equal(state[i] == G.OBS, exp)
    # linear interpolation between keyframes
    mid = 0.5 * (obj["kf"][i, 4, :3].astype(np.float64) + obj["kf"][i, 5, :3])
    b, _ = G.query(obj, np.array([2.25]), dtype=np.float64)
    assert np.allclose(b[i, 0, :3], mid, atol=1e-12)
    # outside the span: finite, clamped to the nearest observed pose
    assert np.array_equal(boxes[i, 0, :3], obj["kf"][i, 3, :3].astype(np.float64))
    # singleton: its pose at every step
    i = tr.index("cone_c")
    assert (state[i] == G.OBS).all()
    assert np.all(boxes[i, :, :3] == obj["kf"][i, 4, :3].astype(np.float64))
    # beyond 5 s: absent + unknown
    b, st, un = G.query(obj, np.array([5.0, 5.1]), return_unknown=True)
    assert (un[:, 1]).all() and not un[:, 0].any() and (st[:, 1] == G.ABSENT_OFFICIAL).all()
    assert st[tr.index("veh_a"), 0] == G.OBS


def test_short_log_horizon():
    frames, _ = make_log(n_frames=7)                   # keyframes 0..3 s only
    obj = G.build_from_frames(frames)
    assert int(obj["n_kf"]) == 7 and G.t_avail(obj) == 3.0
    tr = list(obj["track"])
    b, st, un = G.query(obj, np.array([3.0, 3.1]), return_unknown=True)
    assert st[tr.index("veh_a"), 0] == G.OBS and st[tr.index("veh_a"), 1] == G.ABSENT_OFFICIAL
    assert un[:, 1].all()


def test_duplicate_token_ignored():
    frames, _ = make_log(dup_at=2)
    obj = G.build_from_frames(frames[:17])
    assert int(obj["n_dup"]) == 1
    ref = G.build_from_frames(make_log()[0][:17])
    assert np.array_equal(obj["kf"], ref["kf"])


def test_box_corners_order_matches_sf_common():
    import sf_common as SF
    rng = np.random.default_rng(0)
    b = np.concatenate([rng.normal(size=(20, 3)) * [10, 10, 3], rng.uniform(0.5, 6, size=(20, 2))], 1)
    ref = SF.box_corners(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])
    assert np.allclose(G.box_corners(b), ref, atol=1e-12)


def test_query_torch_matches_numpy(built):
    import torch
    obj, _ = built
    frames, _ = make_log(n_frames=7)
    obj2 = G.build_from_frames(frames)
    t = np.concatenate([np.arange(51) * 0.1, [5.2]])
    P = G.pad_objects([obj, obj2], a_max=None)
    boxes, state = G.query_torch(torch.from_numpy(P["kf"]).double(), torch.from_numpy(P["first"]),
                                 torch.from_numpy(P["meta"]), torch.from_numpy(t),
                                 valid=torch.from_numpy(P["valid"]), n_kf=torch.from_numpy(P["n_kf"]))
    for b, o in enumerate([obj, obj2]):
        bn, sn = G.query(o, t, dtype=np.float64)
        A = bn.shape[0]
        assert np.allclose(boxes[b, :A].numpy(), bn, atol=1e-9)
        assert np.array_equal(state[b, :A].numpy(), sn)
        assert (state[b, A:] == 0).all()
    P2 = G.pad_objects([obj], a_max=2)
    assert P2["kf"].shape[1] == 2 and P2["n_dropped"][0] == len(obj["track"]) - 2


def test_unknown_space(built):
    obj, _ = built
    t = np.array([0.0, 2.0, 4.0, 5.5])
    ego = G.gt_ego_xy(obj, t)
    near = ego[None] + np.array([1.0, 0.5])
    assert not G.unknown_space(obj, near[:, :3], t[:3]).any()
    assert G.unknown_space(obj, near, t)[0, 3], "t > 5 s is unknown"
    # 76 m from the GT ego (and inside R - 10?) -> unknown by reach under both definitions
    far = ego[None, :3] + np.array([0.0, 76.0])
    assert G.unknown_space(obj, far, t[:3], radius_margin=None).all()
    # inside the reach but beyond R - 10 from the origin: unknown only for the extended definition
    R = float(obj["R"])
    d = (R - 5.0) / math.sqrt(2)
    p = np.array([[[d, d]]])
    e0 = G.gt_ego_xy(obj, [4.0])[0]
    if np.hypot(*(p[0, 0] - e0)) < G.REACH_M:
        assert not G.unknown_space(obj, p, [4.0], radius_margin=None).any()
        assert G.unknown_space(obj, p, [4.0]).all()


def test_save_load_roundtrip(tmp_path, built):
    obj, _ = built
    p = tmp_path / "x.npz"
    G.save_objects(p, obj)
    back = G.load_objects(p)
    assert set(back) == set(obj)
    for k in obj:
        assert np.array_equal(back[k], obj[k]), k
    assert back["track"].dtype.kind == "U"
