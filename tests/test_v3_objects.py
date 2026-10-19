"""refiner-v3 B1 tests: causal object sources, pack builder / reader, GT target matching, CA reference, augmentation.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v3_objects.py

Tests marked ``real`` read the BEVFusion caches, the frames parquet and the GT objects (skipped if absent).  Temporary
packs go to D3/test_tmp/<unique>/ and are removed afterwards.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from navsim.agents.para_ssr.ck.temporal import config as TC
from navsim.agents.para_ssr.ck.temporal import objects as O
from navsim.agents.para_ssr.refiner import future_teacher as FT
from navsim.agents.para_ssr.refiner import gt_future as GF
from tools.ck.data import obj_pack as OP

D3 = TC.D3
K = TC.K_OBJ
TEST_TMP = D3 / "test_tmp"
GT_VAL = Path("/workspace/yongjae/SSR/report/53_eval_and_plugin_evidence/A_full16/gt_objects/navtrain_val")
A3_PATH = Path("/workspace/yongjae/SSR/report/51_teacher_specialization_evidence/verify/A_ceiling/scripts/"
               "a3_conflict_features.py")
HAVE_REAL = (FT.TEACHER_ROOTS["navtrain"] / "manifest.json").is_file() and \
    (FT.FUTURE_ROOTS["navtrain"] / "manifest.json").is_file() and (GT_VAL / "index.parquet").is_file()
real = pytest.mark.skipif(not HAVE_REAL, reason="BEVFusion caches / GT objects not available")


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


@pytest.fixture
def tmpdir_d3():
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    p = TEST_TMP / f"t_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    p.mkdir()
    yield p
    shutil.rmtree(p, ignore_errors=True)


@pytest.fixture(scope="module")
def src_nav():
    return O.BevfusionObjectSource("navtrain")


@pytest.fixture(scope="module")
def val_tokens():
    return OP.packed_tokens("navtrain_val")


# ----------------------------------------------------------------------------------------------- conversion
def test_box_conversion_known_answer():
    """heading = wrap(-yaw - pi/2), L = dy, W = dx, positions / velocity through pose_k -> pose0."""
    yaw, x, y, dx, dy, vx, vy = 0.3, 10.0, 2.0, 2.0, 5.0, 3.0, 1.0
    raw = np.array([[x, y, -1.0, dx, dy, 1.5, yaw, vx, vy]], np.float32)
    pose0 = (500100.0, 4400200.0, 0.5)
    pose_k = (500098.0, 4400199.0, 0.45)
    rows = O.teacher_frame_rows(raw, np.array([0.9], np.float32), np.array([0], np.int16), pose_k, pose0, 1.0, 0.2)
    assert rows.shape == (1, 17)
    r = rows[0]
    xr, yr = np.float64(raw[0, 0]), np.float64(raw[0, 1])
    hk, h0 = pose_k[2], pose0[2]
    gx = pose_k[0] + np.cos(hk) * xr - np.sin(hk) * yr
    gy = pose_k[1] + np.sin(hk) * xr + np.cos(hk) * yr
    nx = np.cos(h0) * (gx - pose0[0]) + np.sin(h0) * (gy - pose0[1])
    ny = -np.sin(h0) * (gx - pose0[0]) + np.cos(h0) * (gy - pose0[1])
    h_local = wrap(-np.float64(raw[0, 6]) - np.pi / 2)
    h_n = wrap(h_local + hk - h0)
    dh = hk - h0
    vnx = np.cos(dh) * vx - np.sin(dh) * vy
    vny = np.sin(dh) * vx + np.cos(dh) * vy
    assert abs(r[O.C_X] - nx) < 1e-6 and abs(r[O.C_Y] - ny) < 1e-6
    assert abs(wrap(np.arctan2(r[O.C_SIN], r[O.C_COS]) - h_n)) < 1e-9
    assert r[O.C_L] == pytest.approx(dy) and r[O.C_W] == pytest.approx(dx)
    assert r[O.C_VX] == pytest.approx(vnx, abs=1e-6) and r[O.C_VY] == pytest.approx(vny, abs=1e-6)
    assert r[O.C_SCORE] == pytest.approx(0.9, abs=1e-6) and r[O.C_AGE] == 1.0
    assert r[O.C_CLS0] == 1.0 and r[O.C_CLS0 + 1:O.C_AGE].sum() == 0
    # t0: pose_k = pose0 -> the identity (same chain)
    r0 = O.teacher_frame_rows(raw, np.array([0.9], np.float32), np.array([0], np.int16), pose0, pose0, 0.0, 0.3)[0]
    assert abs(r0[0] - x) < 1e-6 and abs(r0[1] - y) < 1e-6
    assert abs(wrap(np.arctan2(r0[3], r0[2]) - h_local)) < 1e-9
    assert r0[6] == pytest.approx(vx, abs=1e-9) and r0[7] == pytest.approx(vy, abs=1e-9)


def test_threshold_topk_and_ties():
    n = K + 16
    raw = np.zeros((n, 9), np.float32)
    raw[:, 0] = np.arange(n)                       # x = proposal index (identifies rows)
    raw[:, 3:6] = 1.0
    sc = np.full(n, 0.5, np.float32)
    sc[5] = 0.9
    sc[K + 6:] = 0.1                               # below threshold
    sc[3] = np.nan                                 # non-finite score: dropped, counted
    lab = np.zeros(n, np.int16)
    pose = (0.0, 0.0, 0.0)
    rows, info = O.teacher_frame_rows(raw, sc, lab, pose, pose, 0.0, 0.3, return_info=True)
    assert info["n_raw"] == K + 5 and info["n_nonfinite"] == 0    # NaN >= thr is False -> not 'above'
    assert rows.shape == (K, 17)
    assert rows[0, 0] == 5                                          # highest score first
    exp = [i for i in range(K + 6) if i not in (3, 5)][:K - 1]     # ties by proposal index
    assert np.array_equal(rows[1:, 0].astype(int), exp)
    raw[7, 0] = np.inf
    rows2, info2 = O.teacher_frame_rows(raw, sc, lab, pose, pose, 0.0, 0.3, return_info=True)
    assert info2["n_nonfinite"] == 1 and info2["n_raw"] == K + 4 and 7 not in rows2[:, 0].astype(int)
    with pytest.raises(ValueError):
        O.canonical_rows(np.zeros((1, 9)), [0.5], [7], 0.0)


# ----------------------------------------------------------------------------------------------- causal index
class RecordingDict(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.keys_read = []

    def get(self, k, default=None):
        self.keys_read.append(k)
        return super().get(k, default)

    def __getitem__(self, k):
        self.keys_read.append(k)
        return super().__getitem__(k)


def _frames_parquet(path, rows):
    df = pd.DataFrame(rows, columns=["token", "log", "pos", "timestamp", "x", "y", "yaw"])
    df["pos"] = df["pos"].astype("int32")
    df["timestamp"] = df["timestamp"].astype("int64")
    df["lidar2ego_identity"] = True
    df["frame_idx"] = df["pos"]
    df["is_key"] = True
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def planted_index(tmpdir_d3):
    rows = [("tokA00000000000a", "logA", 10, 10_000_000, 0.0, 0.0, 0.0),
            ("tokB00000000000b", "logA", 9, 10_200_000, -1.0, 0.0, 0.0),      # planted: ts > ts0 at pos - 1
            ("tokC00000000000c", "logA", 8, 9_000_000, -2.0, 0.0, 0.0),
            ("tokF00000000000f", "logA", 11, 10_500_000, 1.0, 0.0, 0.0),      # future frame (never read)
            ("tokD00000000000d", "logB", 5, 5_000_000, 0.0, 0.0, 0.0),
            ("tokE00000000000e", "logB", 4, 2_000_000, 0.0, 0.0, 0.0),        # dt = 3 s -> dt_range
            ("tokG00000000000g", "logC", 7, 7_000_000, 0.0, 0.0, 0.1),
            ("tokH00000000000h", "logC", 6, 6_500_000, -3.0, 0.5, 0.1),
            ("tokI00000000000i", "logC", 5, 6_000_000, -6.0, 1.0, 0.1),
            ("tokJ00000000000j", "logC", 8, 7_500_000, 3.0, 0.0, 0.1)]
    ix = O.CausalPastIndex(_frames_parquet(tmpdir_d3 / "frames.parquet", rows))
    ix.bypos = RecordingDict(ix.bypos)
    return ix


def test_causal_index_refuses_planted_and_never_reads_future(planted_index):
    ix = planted_index
    with pytest.raises(O.CausalityError):
        ix.past("tokA00000000000a", 1)
    tok, dt, why = ix.past("tokA00000000000a", 2)
    assert (tok, why) == ("tokC00000000000c", "") and dt == pytest.approx(1.0)
    tok, dt, why = ix.past("tokD00000000000d", 1)
    assert (tok, why) == ("tokE00000000000e", "dt_range") and dt == pytest.approx(3.0)
    tok, dt, why = ix.past("tokD00000000000d", 2)
    assert tok is None and why == "not_in_parquet" and np.isnan(dt)
    for j in (0, -1):
        with pytest.raises(ValueError):
            ix.past("tokA00000000000a", j)
    pos0 = {"logA": 10, "logB": 5, "logC": 7}
    assert ix.bypos.keys_read and all(p < pos0[lg] for lg, p in ix.bypos.keys_read)


class FakeCache:
    def __init__(self, boxes):
        self.boxes = boxes
        self.reads = []

    def has(self, t):
        return t in self.boxes

    def load_boxes(self, t):
        self.reads.append(t)
        return self.boxes[t]


def _raw(n, x0=5.0, score=0.9, label=0, vx=2.0):
    b = np.zeros((200, 9), np.float32)
    b[:n, 0] = x0 + np.arange(n)
    b[:n, 3], b[:n, 4], b[:n, 5] = 2.0, 4.5, 1.5
    b[:n, 7] = vx
    s = np.zeros(200, np.float32)
    s[:n] = score
    lab = np.zeros(200, np.int16)
    lab[:n] = label
    return dict(pred_boxes_3d=b, pred_scores_3d=s, pred_labels_3d=lab)


def test_source_frames_order_dt_and_causality(planted_index):
    src = O.BevfusionObjectSource.__new__(O.BevfusionObjectSource)
    src.index = planted_index
    cur = FakeCache({"tokG00000000000g": _raw(3), "tokH00000000000h": _raw(2), "tokA00000000000a": _raw(1),
                     "tokC00000000000c": _raw(1), "tokB00000000000b": _raw(1)})
    fut = FakeCache({"tokH00000000000h": _raw(5), "tokI00000000000i": _raw(4), "tokJ00000000000j": _raw(9)})
    src.current, src.future = cur, fut
    fr = src.frames("tokG00000000000g")
    assert fr["obj"].shape == (3, K, 17) and fr["obj"].dtype == np.float32
    assert fr["valid"].tolist() == [True, True, True]
    assert fr["src"].tolist() == [1, 1, 2]                          # current first (H in both -> current), then future
    assert fr["mask"].sum(1).tolist() == [3, 2, 4]
    assert np.allclose(fr["dt"], [0.0, 0.5, 1.0]) and np.allclose(fr["obj"][1, :2, 16], 0.5)
    assert "tokJ00000000000j" not in cur.reads + fut.reads           # the pos + 1 frame is never read
    with pytest.raises(O.CausalityError):
        src.frames("tokA00000000000a")
    fd = src.frames("tokD00000000000d")
    assert fd["valid"].tolist() == [False, False, False] and fd["why"].tolist() == ["no_npz", "dt_range", "not_in_parquet"]
    assert np.isnan(fd["dt"]).all() and not fd["mask"].any()


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        O.V2DetObjectSource()
    with pytest.raises(NotImplementedError):
        O.GtObjectSource()


# ----------------------------------------------------------------------------------------------- targets / CA
def _gt(n_kf=11):
    A = 3
    kf = np.zeros((A, 11, 6), np.float32)
    t = 0.5 * np.arange(11)
    kf[0, :, 0], kf[0, :, 1], kf[0, :, 3], kf[0, :, 5] = 10 + 5 * t, 0.0, 5.0, 1.0       # vehicle, every kf
    kf[1, :5, 0], kf[1, :5, 1], kf[1, :5, 5] = 5.0, 5.0 + t[:5], 1.0                       # pedestrian k 0..4
    kf[2, 0, 0], kf[2, 0, 1], kf[2, 0, 5] = 3.0, -3.0, 1.0                                 # static cone, singleton
    first = np.array([[4.5, 2.0, 0, 5, 0, 0], [0.5, 0.5, 0, 0, 1, 0], [0.3, 0.3, 0, 0, 0, 0]], np.float32)
    meta = np.array([[0, 1, 0, 10, 0], [1, 1, 0, 4, 0], [3, 0, 0, 0, 1]], np.int16)
    for i in range(A):
        meta[i, 3] = min(meta[i, 3], n_kf - 1)
    return dict(kf=kf, first=first, meta=meta, track=np.array(["v", "p", "c"], "<U16"), n_kf=np.int32(n_kf),
                R=np.float32(80.0))


def _rows(spec):
    """spec: list of (x, y, cls) -> obj0 [K, 17], mask0 [K]."""
    obj = np.zeros((K, 17), np.float32)
    mask = np.zeros(K, bool)
    for i, (x, y, c) in enumerate(spec):
        obj[i, 0], obj[i, 1], obj[i, 2], obj[i, 9 + c] = x, y, 1.0, 1.0
        mask[i] = True
    return obj, mask


def test_match_targets_known_answer():
    obj, mask = _rows([(10.5, 0.1, 0), (5.2, 5.0, 1), (3.0, -3.0, 3), (30.0, 0.0, 0), (5.0, 5.0, 3)])
    xyh, st, tr, s = O.match_targets(obj, mask, _gt(), return_stats=True)
    assert xyh.shape == (K, 10, 3) and xyh.dtype == np.float32 and st.dtype == np.int8 and tr.dtype == np.int32
    assert st[0].tolist() == [1] * 10 and tr[0] == 0
    assert np.allclose(xyh[0, :, 0], 10 + 5 * 0.5 * np.arange(1, 11)) and np.allclose(xyh[0, :, 1], 0)
    assert st[1].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2] and tr[1] == 1                # OBS while t <= 2.0 s
    assert st[2].tolist() == [0] * 10 and tr[2] == -1                                     # static: no target
    assert st[3].tolist() == [0] * 10 and tr[3] == -1                                     # unmatched
    assert st[4].tolist() == [0] * 10                                                     # static slot never takes an agent
    assert (st[5:] == 0).all() and (tr[5:] == -1).all()
    assert s["n_agent"] == 3 and s["n_matched"] == 2 and s["n_static_matched"] == 1 and s["n_cross"] == 0
    # log ends early: t_avail = 4.0 s -> steps 9, 10 BEYOND
    _, st2, _ = O.match_targets(obj, mask, _gt(n_kf=9))
    assert st2[0].tolist() == [1] * 8 + [3, 3]
    # VRU detection on a vehicle track: allowed (same agent group) and counted
    obj3, mask3 = _rows([(10.2, 0.0, 2)])
    _, st3, _, s3 = O.match_targets(obj3, mask3, _gt(), return_stats=True)
    assert st3[0, 0] == 1 and s3["n_cross"] == 1
    # gate 2.0 m
    obj4, mask4 = _rows([(12.5, 0.0, 0)])
    _, st4, _ = O.match_targets(obj4, mask4, _gt())
    assert (st4 == 0).all()


def test_ca_forecast_np_synthetic():
    obj = np.zeros((3, K, 17))
    mask = np.zeros((3, K), bool)
    # slot 0: vehicle at (10, 0) with v 10 m/s along x; t-1.0 box predicted at (0, 0) with v 12 -> a = -2
    obj[0, 0, [0, 1, 2, 6, 9]] = [10.0, 0.0, 1.0, 10.0, 1.0]
    obj[2, 0, [0, 1, 2, 6, 9, 16]] = [0.2, 0.1, 1.0, 12.0, 1.0, 1.0]
    # slot 1: pedestrian without history (CV); slot 2: static cone with a velocity (held)
    obj[0, 1, [0, 1, 7, 10]] = [5.0, 5.0, 1.0, 1.0]
    obj[0, 2, [0, 1, 6, 12]] = [3.0, -3.0, 0.4, 1.0]
    mask[0, :3] = True
    mask[2, 0] = True
    valid = np.array([True, False, True])
    ca = O.ca_forecast_np(obj, mask, valid)
    t = np.asarray(TC.FC_T)
    assert ca["has"].tolist()[:3] == [True, False, False] and ca["acc"][0] == pytest.approx(-2.0)
    te = np.minimum(t, 5.0)
    assert np.allclose(ca["xy"][0, :, 0], 10 + 10 * te - te ** 2) and np.allclose(ca["xy"][0, :, 1], 0)
    assert np.allclose(ca["xy"][1, :, 1], 5 + t) and np.allclose(ca["xy"][1, :, 0], 5)
    assert np.allclose(ca["xy"][2], [3.0, -3.0])
    assert np.allclose(ca["xy"][3:], 0)
    # history gate 1.5 m and the 'prefer t-1.0' rule
    obj2 = obj.copy()
    obj2[2, 0, 0] = 2.0                                               # 2 m from the prediction -> no association
    assert not O.ca_forecast_np(obj2, mask, valid)["has"][0]
    # slow objects (|v0| <= 0.5) -> a = 0 even with history
    obj3 = obj.copy()
    obj3[0, 0, 6] = 0.4
    obj3[2, 0, 0], obj3[2, 0, 6] = 9.6, 3.0
    c3 = O.ca_forecast_np(obj3, mask, valid)
    assert c3["has"][0] and c3["acc"][0] == 0.0


# ----------------------------------------------------------------------------------------------- augmentation
def _item(seed=0):
    rng = np.random.default_rng(seed)
    obj = rng.normal(size=(3, K, 17)).astype(np.float32)
    mask = np.zeros((3, K), bool)
    mask[:, :20] = True
    obj[~mask] = 0
    return dict(obj_in=obj, obj_mask=mask, frame_valid=np.array([True, True, True]), obj_ok=np.array(True),
                fc_xyh=rng.normal(size=(K, 10, 3)).astype(np.float32), fc_state=np.ones((K, 10), np.int8))


def test_augment_determinism_and_rule(monkeypatch):
    cfg = TC.TemporalCfg(on=True, forecaster="ca", n_modes=1)
    it = _item()
    snap = {k: v.copy() for k, v in it.items()}
    a = OP.augment_objects(it, "abcdef0123456789", 3, 7, cfg)
    b = OP.augment_objects(it, "abcdef0123456789", 3, 7, cfg.to_dict())
    for k in a:
        assert a[k].tobytes() == b[k].tobytes()
    for k in it:                                                    # input untouched
        assert it[k].tobytes() == snap[k].tobytes()
    c = OP.augment_objects(it, "abcdef0123456789", 4, 7, cfg)
    assert a["obj_in"].tobytes() != c["obj_in"].tobytes()
    # expected values from the pinned draw order
    rng = OP.token_rng("abcdef0123456789", 3, 7, "objaug")
    uf, uo, nz = rng.random(2), rng.random((3, K)), rng.standard_normal((3, K, 2)) * 0.3
    fv = np.array([True, uf[0] >= 0.1, uf[1] >= 0.1])
    m = it["obj_mask"] & fv[:, None] & ~(uo < 0.05)
    assert np.array_equal(a["frame_valid"], fv) and np.array_equal(a["obj_mask"], m)
    exp = it["obj_in"].copy()
    exp[..., 6:8] += np.where(m[..., None], nz, 0.0).astype(np.float32)
    exp[~m] = 0
    assert np.array_equal(a["obj_in"], exp)
    assert a["fc_xyh"] is it["fc_xyh"] and a["fc_state"] is it["fc_state"]    # targets untouched
    # off -> same object, no draw at all
    def boom(*a, **k):
        raise AssertionError("token_rng drawn while temporal is off")
    monkeypatch.setattr(OP, "token_rng", boom)
    assert OP.augment_objects(it, "abcdef0123456789", 3, 7, TC.TemporalCfg()) is it
    assert OP.augment_objects(it, "abcdef0123456789", 3, 7, None) is it


def test_augment_rates():
    cfg = TC.TemporalCfg(on=True, forecaster="ca", n_modes=1)
    it = _item()
    it["obj_mask"][:] = True
    drops, fdrop, vn = [], [], []
    for e in range(400):
        a = OP.augment_objects(it, f"{e:016x}", 0, 0, cfg)
        fdrop.append(~a["frame_valid"][1:])
        drops.append(~a["obj_mask"][0])
        d = (a["obj_in"][0, :, 6:8] - it["obj_in"][0, :, 6:8])[a["obj_mask"][0]]
        vn.append(d.ravel())
    assert abs(np.mean(fdrop) - 0.1) < 0.03 and abs(np.mean(drops) - 0.05) < 0.01
    assert abs(np.concatenate(vn).std() - 0.3) < 0.01


# ----------------------------------------------------------------------------------------------- pack reader
def _write_pack(P, toks, targets=True, version=OP.PACK_VERSION, status="complete"):
    P.mkdir(parents=True, exist_ok=True)
    n = len(toks)
    specs = {**OP.PACK_FILES, **(OP.TARGET_FILES if targets else {})}
    for k, (dt, shp) in specs.items():
        np.save(P / f"{k}.npy", np.zeros((n,) + tuple(shp), dt))
    pd.DataFrame({"token": toks, "row": np.arange(n)}).to_parquet(P / "tokens.parquet", index=False)
    (P / "build.json").write_text(json.dumps(dict(version=version, source="bevfusion", split=P.name, targets=targets,
                                                  status=status, n=n)))


def test_objpack_checks(tmpdir_d3):
    toks = np.array([f"{i:016x}" for i in range(5)])
    root = tmpdir_d3 / "objpack"
    P = root / "bevfusion" / "navtrain_val"
    _write_pack(P, toks)
    pk = OP.ObjPack("navtrain_val", root=root, expect_tokens=toks)
    it = pk.item(2)
    assert set(it) == {"obj_in", "obj_mask", "frame_valid", "obj_ok", "fc_xyh", "fc_state"}
    assert set(OP.ObjPack("navtrain_val", root=root, expect_tokens=toks, targets=False).item(0)) == set(TC.OBJ_INPUT_KEYS) | {"obj_ok"}
    with pytest.raises(IndexError):
        pk.item(5)
    with pytest.raises(ValueError):
        OP.ObjPack("navtrain_val", root=root, expect_tokens=toks[::-1])
    with pytest.raises(ValueError):                                  # default: packed order prefix -> differs
        OP.ObjPack("navtrain_val", root=root)
    P2 = root / "bevfusion" / "navtest"
    _write_pack(P2, toks, targets=False)
    with pytest.raises(ValueError):
        OP.ObjPack("navtest", root=root, expect_tokens=toks, targets=True)
    assert not OP.ObjPack("navtest", root=root, expect_tokens=toks, targets=False).has_targets
    _write_pack(P2, toks, targets=False, version="v0")
    with pytest.raises(ValueError):
        OP.ObjPack("navtest", root=root, expect_tokens=toks, targets=False)
    _write_pack(P2, toks, targets=False, status="failed_truncation")
    with pytest.raises(ValueError):
        OP.ObjPack("navtest", root=root, expect_tokens=toks, targets=False)


# ----------------------------------------------------------------------------------------------- real data
@real
def test_pack_build_real_32(tmpdir_d3, val_tokens):
    """32 real val tokens through the builder: shapes, dtypes, token order, resumability, t0 rows == TeacherCache."""
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    from tools.ck.v3 import build_obj_pack as B

    root = tmpdir_d3 / "objpack"
    args = ["--split", "navtrain_val", "--workers", "2", "--limit", "32", "--chunk", "16", "--out-root", str(root)]
    assert B.main(args + ["--keep-parts"]) == 0
    P = root / "bevfusion" / "navtrain_val"
    pk = OP.ObjPack("navtrain_val", root=root)
    assert np.array_equal(pk.tokens, val_tokens[:32])
    first = {k: np.load(P / f"{k}.npy").copy() for k in list(OP.PACK_FILES) + list(OP.TARGET_FILES)}
    for k, (dt, shp) in {**OP.PACK_FILES, **OP.TARGET_FILES}.items():
        assert first[k].shape == (32,) + tuple(shp) and first[k].dtype == np.dtype(dt), k
    assert first["obj_ok"].all() and first["frame_valid"][:, 0].all()
    assert np.array_equal(np.isnan(first["frame_dt"]), ~first["frame_valid"])
    assert (first["obj"][~first["obj_mask"]] == 0).all()
    with pytest.raises(SystemExit):                                   # a complete pack is never overwritten
        B.main(args)
    # resume: drop the final files, keep the parts -> the same bytes
    for f in P.iterdir():
        if f.is_file():
            f.unlink()
    assert B.main(args) == 0
    for k, v in first.items():
        assert np.load(P / f"{k}.npy").tobytes() == v.tobytes(), k
    assert not (P / "_parts").exists()
    # t0 rows vs data.TeacherCache.load_det (the CK2 DET teacher's own reader) + the raw conversion
    tc = TeacherCache.for_subset("navtrain")
    for r in range(32):
        d = tc.load_det(val_tokens[r], s_grid=False)
        rows = O.teacher_frame_rows(d["pred_boxes_3d"], d["pred_scores_3d"], d["pred_labels_3d"], (0, 0, 0), (0, 0, 0),
                                    0.0, TC.SCORE_THR[0])
        m = first["obj_mask"][r, 0]
        assert m.sum() == len(rows)
        assert np.allclose(first["obj"][r, 0, m], rows.astype(np.float32), atol=1e-4)
    # past-frame dt from the frames parquet timestamps
    ix = O.CausalPastIndex(FT.INDEX_ROOT / "future_frames_train_10f.parquet")
    for r in range(32):
        _, _, ts0, _ = ix.t0(val_tokens[r])
        for f in (1, 2):
            if first["frame_valid"][r, f]:
                ts = ix.rows[str(first["frame_token"][r, f])][2]
                assert first["frame_dt"][r, f] == pytest.approx((ts0 - ts) * 1e-6, abs=1e-6)
                assert np.allclose(first["obj"][r, f, first["obj_mask"][r, f], 16], first["frame_dt"][r, f])


@real
def test_heading_and_size_vs_gt_real(src_nav, val_tokens):
    """60 val tokens, matched vehicles at t0: median |heading error| < 0.02 rad; |median(L - L_gt)| < 0.05 m (signed:
    the critic's 0.003 m was a signed median; the absolute median is detector size noise, 0.086 m [실측 2026-10-11]);
    the swapped convention (dx as L) is off by metres."""
    eh, el, ew, sw = [], [], [], []
    for tok in val_tokens[::300][:60]:
        fr = src_nav.frames(tok, dtype=np.float64)
        gt = GF.load_objects(GT_VAL / f"{tok}.npz")
        _, st, tr = O.match_targets(fr["obj"][0], fr["mask"][0], gt)
        bz, _ = GF.query(gt, np.zeros(1), dtype=np.float64)
        for i in np.flatnonzero(tr >= 0):
            if O.row_classes(fr["obj"][0, i]) != 0:
                continue
            h = np.arctan2(fr["obj"][0, i, 3], fr["obj"][0, i, 2])
            eh.append(abs(wrap(h - bz[tr[i], 0, 2])))
            el.append(fr["obj"][0, i, 4] - gt["first"][tr[i], GF.FI_L])
            ew.append(fr["obj"][0, i, 5] - gt["first"][tr[i], GF.FI_W])
            sw.append(abs(fr["obj"][0, i, 5] - gt["first"][tr[i], GF.FI_L]))
    assert len(eh) > 100
    assert np.median(eh) < 0.02, np.median(eh)
    assert abs(np.median(el)) < 0.05 and abs(np.median(ew)) < 0.05, (np.median(el), np.median(ew))
    assert np.median(np.abs(el)) < 0.15 and np.median(sw) > 1.0, (np.median(np.abs(el)), np.median(sw))


@real
def test_past_frame_parked_vehicles_real(src_nav, val_tokens):
    """t-1.0 detections of parked vehicles moved into the t0 N frame land on the GT t0 boxes (median < 0.2 m)."""
    d = []
    for tok in val_tokens[::97]:
        fr = src_nav.frames(tok, dtype=np.float64)
        if not fr["valid"][2]:
            continue
        gt = GF.load_objects(GT_VAL / f"{tok}.npz")
        if not len(gt["track"]):
            continue
        bz, sz = GF.query(gt, np.zeros(1), dtype=np.float64)
        kf = gt["kf"]
        parked = (sz[:, 0] == GF.OBS) & (gt["meta"][:, GF.MT_CLASS] == 0) & (kf[:, 0, GF.KF_PRESENT] == 1) & \
            (np.hypot(kf[:, 0, GF.KF_VX], kf[:, 0, GF.KF_VY]) < 0.1) & (kf[:, 2, GF.KF_PRESENT] == 1)
        m = fr["mask"][2] & (O.row_classes(fr["obj"][2]) == 0)
        if not parked.any() or not m.any():
            continue
        p = fr["obj"][2, m, :2]
        for g in np.flatnonzero(parked):
            dd = np.hypot(p[:, 0] - bz[g, 0, 0], p[:, 1] - bz[g, 0, 1]).min()
            if dd < 2.0:
                d.append(dd)
        if len(d) >= 400:
            break
    assert len(d) >= 50
    assert np.median(d) < 0.2, np.median(d)


def _load_a3():
    old_path, old_dwb = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("a3_conflict_features_ref", str(A3_PATH))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_dwb
    assert mod.FT.__file__ == FT.__file__                           # resolved to this worktree's copy
    return mod


@real
@pytest.mark.skipif(not A3_PATH.is_file(), reason="a3 script not available")
def test_ca_equals_a3_real(src_nav, val_tokens):
    """ca_forecast_np == a3 Worker.history + world('CA') on agents of 50 real val tokens (n_raw <= K_OBJ, no dt_range)."""
    A3 = _load_a3()
    w = A3.Worker("val")
    n_tok, n_obj, n_hist, worst = 0, 0, 0, 0.0
    for tok in val_tokens[7::131]:
        fr = src_nav.frames(tok, dtype=np.float64)
        if (fr["n_raw"] > TC.K_OBJ).any() or (fr["why"] == "dt_range").any():
            continue
        b0, l0 = w.boxes(tok)
        if not np.isin(l0, A3.AGENT).any():
            continue
        acc, has, _ = w.history(tok, b0, l0)
        c, *_ = w.world("CA", np.asarray(TC.FC_T), b0, l0, acc, None)           # [10, N, 2]
        ours = O.ca_forecast_np(fr["obj"], fr["mask"], fr["valid"])
        d = src_nav.current.load_boxes(tok)
        _, info = O.teacher_frame_rows(d["pred_boxes_3d"], d["pred_scores_3d"], d["pred_labels_3d"], (0, 0, 0),
                                       (0, 0, 0), 0.0, TC.SCORE_THR[0], return_info=True)
        a3_idx = np.flatnonzero(d["pred_scores_3d"] >= A3.THR)
        pos = {int(p): i for i, p in enumerate(a3_idx)}
        for s, p in enumerate(info["idx"]):
            i = pos[int(p)]
            if l0[i] not in (0, 1, 2):
                continue
            assert ours["agent"][s]
            assert bool(ours["has"][s]) == bool(has[i])
            assert abs(ours["acc"][s] - acc[i]) < 1e-9
            worst = max(worst, float(np.abs(ours["xy"][s] - c[:, i]).max()))
            n_obj += 1
            n_hist += int(has[i])
        n_tok += 1
        if n_tok >= 50:
            break
    assert n_tok >= 50 and n_obj > 200 and n_hist > 50
    assert worst < 1e-9, worst


@real
def test_match_rate_val_full_pack():
    """match rate of the full navtrain_val pack within 0.5 pp of f4's 99.38 % (87,918 agent boxes)."""
    bj = TC.PACK_ROOT / "bevfusion" / "navtrain_val" / "build.json"
    if not bj.is_file() or json.loads(bj.read_text()).get("status") != "complete":
        pytest.skip("full navtrain_val pack not built yet")
    b = json.loads(bj.read_text())
    assert abs(b["counts"]["targets"]["match_rate"] - 0.9938) < 0.005
    assert b["counts"]["n_errors"] == 0 and b["counts"]["obj_ok_false"] == 0
