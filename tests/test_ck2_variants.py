"""CK2 candidate variants (navsim/agents/para_ssr/ck/variants.py): table / identity / validity / determinism on
synthetic + anchor trajectories, and on real navtrain_train candidates equality with the verified reference
/home/external-user/ssd/yongjae_refiner/ck/analysis/variants_cf/cf_variants.py (trajectories <= 1e-5 on its 20
unit-test tokens; official labels of all 12 'cross' variants on 8 tokens equal the stored navtrain_train/cf.npz).

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_variants.py

The scorer test (official labels, 8 tokens x 192 trajectories, 8 worker processes, ~1 min) is skipped unless
CK2_SCORER_TEST=1.  The reference module is loaded by path with bytecode writing off and sys.path restored, after
this worktree's navsim / tools.ck.data are imported (so its 'sys.path.insert(0, SSR-ck)' cannot pull SSR-ck code).
"""
import importlib.util
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import variants as VV  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import S_LAT_MIN  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
CFV_DIR = CK_DATA / "analysis/variants_cf"
SPLIT = "navtrain_train"
ANCHORS = Path("/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy")
OLD_CK = "/workspace/yongjae/SSR-ck/"            # stopped run's worktree: nothing may be imported from it
N_LAB_TOK, N_LAB_WORKERS = 8, 8


def _synthetic(T=3, K=5, seed=0):
    """Smooth forward trajectories of mixed speed / curvature, incl. a stopped and a 2 m (lateral-disabled) one."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    out = []
    for _ in range(T * K):
        v = float(torch.rand(1, generator=g)) * 15.0
        acc = (float(torch.rand(1, generator=g)) - 0.5) * 2.0
        kap = (0.001 + 0.024 * float(torch.rand(1, generator=g))) * (1 if float(torch.rand(1, generator=g)) < .5 else -1)
        s = torch.cummax(torch.clamp(v * t + 0.5 * acc * t * t, min=0.0), 0).values
        h = kap * s
        out.append(torch.stack([torch.sin(h) / kap, (1 - torch.cos(h)) / kap, h], -1))
    tau = torch.stack(out).reshape(T, K, 8, 3).float()
    tau[0, 0] = 0.0                                                     # stopped
    tau[0, 1, :, 0] = torch.linspace(0.25, 2.0, 8)                      # S_8 = 2 m < S_LAT_MIN
    tau[0, 1, :, 1:] = 0.0
    v0 = torch.rand(T, generator=g) * 10.0
    return tau, v0


def _anchors():
    if not ANCHORS.is_file():
        pytest.skip("anchors file absent")
    return torch.as_tensor(np.load(ANCHORS), dtype=torch.float32)        # [256, 8, 3]


# ----------------------------------------------------------------------------------------------- unit
def test_table_names_and_prefix():
    sep = VV.variant_table(combine="separate")
    cro = VV.variant_table(combine="cross")
    assert sep == [(0.0, 0.0), (-1.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    assert len(cro) == 12 and cro[:6] == sep and set(cro[6:]) == {(a, d) for a in VV.SPEEDS for d in VV.LATS}
    assert [VV.variant_name(*x) for x in sep] == ["id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5"]
    for bad in (dict(speeds=(0.0,)), dict(speeds=(2.0,)), dict(lats=(0.0,)), dict(lats=(0.5, 0.5)),
                dict(combine="both")):
        with pytest.raises(ValueError):
            VV.variant_table(**bad)


@pytest.mark.parametrize("combine", ["separate", "cross"])
def test_identity_shapes_valid(combine):
    tau, v0 = _synthetic()
    o = VV.make_variants(tau, v0, combine=combine)
    V = 6 if combine == "separate" else 12
    T, K = tau.shape[:2]
    assert o["traj"].shape == (T, K, V, 8, 3) and o["traj"].dtype == torch.float32
    assert o["valid"].shape == (T, K, V) and o["valid"].dtype == torch.bool
    assert torch.equal(o["traj"][:, :, 0], tau)                                       # identity: bitwise
    assert torch.isfinite(o["traj"]).all()
    tab = VV.variant_table(combine=combine)
    assert o["speed"].tolist() == [a for a, _ in tab] and o["lat"].tolist() == [d for _, d in tab]
    S8 = o["meta"]["S8"]
    has_lat = o["lat"] != 0
    assert torch.equal(o["valid"], ~(has_lat[None, None] & (S8 < S_LAT_MIN)[..., None]))
    assert not o["valid"][0, 1, has_lat].any() and o["valid"][0, 1, ~has_lat].all()
    # lateral-disabled candidate: lateral-only variants == identity, combos == their speed-only parent
    for v, (a, d) in enumerate(tab):
        if d == 0:
            continue
        parent = tab.index((a, 0.0))
        assert torch.equal(o["traj"][0, 1, v], o["traj"][0, 1, parent]), (v, parent)
    # lateral variants keep the knot timing roughly and shift toward the requested side on long candidates
    long_ = S8 >= 5.0
    for v, (a, d) in enumerate(tab):
        if a == 0 and d != 0:
            de = o["meta"]["d_end"][..., v][long_]
            assert (torch.sign(de) == np.sign(d)).all() and (de.abs() <= 1.05 * abs(d)).all()
    # identity meta
    m = o["meta"]
    assert (m["alpha"][..., 0] == 1).all() and (m["beta"][..., 0] == 1).all() and (m["dev_xy"][..., 0] == 0).all()
    assert m["names"][0] == "id" and m["mode"][0] == "id"


def test_speed_direction_on_anchors():
    A = _anchors()
    tau = A.reshape(8, 32, 8, 3)
    v0 = torch.linspace(0.0, 15.0, 8)
    o = VV.make_variants(tau, v0, combine="separate")
    ds4 = o["meta"]["ds4"]
    names = o["meta"]["names"]
    assert (ds4[..., names.index("a-1.0")] <= ds4[..., names.index("a-0.5")] + 1e-6).all()
    assert (ds4[..., names.index("a-0.5")] <= 1e-6).all()
    assert (ds4[..., names.index("a+0.5")] >= -1e-6).all()
    assert torch.equal(o["traj"][:, :, 0], tau)


def test_deterministic_chunking_and_numpy_wrapper():
    A = _anchors()
    tau = A[:96].reshape(3, 32, 8, 3)
    v0 = torch.tensor([0.0, 5.0, float("nan")])
    o1 = VV.make_variants(tau, v0, combine="cross")
    o2 = VV.make_variants(tau, v0, combine="cross", max_rows=37)                        # different decode batches
    o3 = VV.make_variants(tau, v0, combine="cross")
    assert torch.equal(o1["traj"], o3["traj"])
    assert (o1["traj"] - o2["traj"]).abs().max() <= 1e-6
    sep = VV.make_variants(tau, v0, combine="separate")
    assert (sep["traj"] - o1["traj"][:, :, :6]).abs().max() <= 1e-6                     # separate = prefix of cross
    assert torch.equal(sep["valid"], o1["valid"][:, :, :6])
    # NaN v0 -> 0
    o0 = VV.make_variants(tau[2:], torch.tensor([0.0]), combine="cross")
    assert torch.equal(o0["traj"], o1["traj"][2:])
    # numpy wrapper: batched and single token
    n = VV.make_variants_np(tau.numpy(), v0.numpy(), combine="cross")
    assert np.array_equal(n["traj"], o1["traj"].numpy()) and np.array_equal(n["valid"], o1["valid"].numpy())
    s = VV.make_variants_np(tau[1].numpy(), 5.0, combine="cross")
    assert s["traj"].shape == (32, 12, 8, 3) and s["meta"]["S8"].shape == (32,)
    assert np.array_equal(s["traj"], o1["traj"][1].numpy())
    assert np.array_equal(s["speed"], o1["speed"].numpy()) and s["meta"]["names"] == o1["meta"]["names"]
    # per-candidate v0 [T, K]
    pk = VV.make_variants(tau, torch.tensor([0.0, 5.0, 0.0])[:, None].expand(3, 32), combine="cross")
    assert torch.equal(pk["traj"], o1["traj"])


# ----------------------------------------------------------------------------------------------- reference (real data)
def _need_real():
    need = [CFV_DIR / "cf_variants.py", CFV_DIR / "unit_test/cf20.npz", CK_DATA / f"packed/{SPLIT}/cand.npy",
            CK_DATA / f"packed/{SPLIT}/status.npy", CK_DATA / f"packed/{SPLIT}/tokens.parquet"]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        pytest.skip(f"real data absent: {miss}")


def _load_ref():
    """cf_variants loaded by path; this worktree's modules are imported first and sys.path is restored."""
    from tools.ck.data import label_cands as LC   # pins navsim to this worktree; the reference reuses this module
    path0, dwb = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True                # never write into analysis/variants_cf/__pycache__
    try:
        spec = importlib.util.spec_from_file_location("cf_variants_ref", CFV_DIR / "cf_variants.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dwb
        sys.path[:] = path0                       # drop its sys.path.insert(0, SSR-ck)
    leaked = sorted(n for n, m in list(sys.modules.items())
                    if str(getattr(m, "__file__", "") or "").startswith(OLD_CK))
    assert not leaked, leaked
    assert mod.LC is LC and mod.decode is VV.decode
    return mod


def _v0_of(st):
    st = np.asarray(st, np.float64)
    return float(np.nan_to_num(np.hypot(st[4], st[5]), nan=0.0))     # = cf_variants.score_chunk


def _ref_index(R, a, d):
    return R.SPEEDS.index(a), R.LATS.index("none" if d == 0 else f"r{d:+.1f}")


def test_matches_cf_variants_trajectories():
    _need_real()
    R = _load_ref()
    u = np.load(CFV_DIR / "unit_test/cf20.npz")
    rows = u["rows"]
    cand = np.load(CK_DATA / f"packed/{SPLIT}/cand.npy", mmap_mode="r")
    status = np.load(CK_DATA / f"packed/{SPLIT}/status.npy", mmap_mode="r")
    teach = np.load(R.TEACH[SPLIT][0], mmap_mode="r")
    tau = np.stack([np.asarray(cand[r], np.float32) for r in rows])              # [20, 16, 8, 3]
    v0 = np.array([_v0_of(status[r]) for r in rows])
    ref, dec = [], []
    for i, r in enumerate(rows):
        tr, de, _ = R.build_variants(tau[i], np.asarray(teach[r], np.float32), v0[i])
        ref.append(tr)
        dec.append(de)
    ref, dec = np.stack(ref), np.stack(dec)                                      # [20,16,5,6,8,3], [20,16,5,6,16]
    DC = {c: i for i, c in enumerate(R.DEC_COLS)}
    o = VV.make_variants_np(tau, v0, combine="cross")
    tab = VV.variant_table(combine="cross")
    devs = []
    for v, (a, d) in enumerate(tab):
        si, li = _ref_index(R, a, d)
        dv = np.abs(o["traj"][:, :, v] - ref[:, :, si, li]).max()
        devs.append(float(dv))
        assert dv <= 1e-5, (VV.variant_name(a, d), dv)
        on = dec[:, :, si, li, DC["lat_on"]] > 0
        if d != 0:
            assert np.array_equal(o["valid"][:, :, v], on)
        else:
            assert o["valid"][:, :, v].all()
        if v:
            for k in ("alpha", "beta"):
                assert np.abs(o["meta"][k][:, :, v] - dec[:, :, si, li, DC[k]]).max() <= 1e-5, k
            assert np.abs(o["meta"]["ds4"][:, :, v] - dec[:, :, si, li, DC["ds4"]]).max() <= 1e-4
    assert np.array_equal(o["traj"][:, :, 0], tau)
    n_exact = sum(np.array_equal(o["traj"][:, :, v], ref[:, :, _ref_index(R, a, d)[0], _ref_index(R, a, d)[1]])
                  for v, (a, d) in enumerate(tab))
    print(f"\n[cf_variants parity] 20 tokens x 16 cand: max|dev| per variant {np.round(devs, 9).tolist()}, "
          f"bitwise-equal variants {n_exact}/{len(tab)}, lateral-disabled cand {int((~o['meta']['lat_on']).sum())}")


# ----------------------------------------------------------------------------------------------- official labels
_SW = {}


def _score_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from tools.ck.data import label_cands as LC
    _SW["LC"] = LC
    LC.ST.get_simulator_scorer()


def _score_one(task):
    tok, log, trajs = task
    LC = _SW["LC"]
    mc = LC.ST.load_metric_cache(LC.CM.mc_path(SPLIT, log, tok))
    res = LC.ST.score_token(mc, trajs)
    return np.array([[r[c] for c in LC.CM.LABEL_COLS] for r in res], np.float64)


@pytest.mark.skipif(os.environ.get("CK2_SCORER_TEST") != "1", reason="official scorer test: set CK2_SCORER_TEST=1")
def test_official_labels_match_cf_npz():
    _need_real()
    import pandas as pd
    from tools.ck.data import label_cands as LC
    cf_path = CFV_DIR / f"{SPLIT}/cf.npz"
    if not cf_path.is_file():
        pytest.skip("cf.npz absent")
    z = np.load(cf_path)
    toks, rows = z["tokens"][:N_LAB_TOK], z["rows"][:N_LAB_TOK]
    lab_ref = z["lab"][:N_LAB_TOK].astype(np.float64)                          # [n,16,5,6,9]
    speeds, lats = list(z["speeds"]), [str(x) for x in z["lats"]]
    assert tuple(str(c) for c in z["label_cols"]) == tuple(LC.CM.LABEL_COLS)
    del z
    tp = pd.read_parquet(CK_DATA / f"packed/{SPLIT}/tokens.parquet")
    assert list(tp.token.values[rows]) == list(toks)
    cand = np.load(CK_DATA / f"packed/{SPLIT}/cand.npy", mmap_mode="r")
    status = np.load(CK_DATA / f"packed/{SPLIT}/status.npy", mmap_mode="r")
    tau = np.stack([np.asarray(cand[r], np.float32) for r in rows])
    v0 = np.array([_v0_of(status[r]) for r in rows])
    o = VV.make_variants_np(tau, v0, combine="cross")
    V = o["traj"].shape[2]
    tasks = [(str(toks[i]), str(tp.log.values[rows[i]]), o["traj"][i].reshape(-1, 8, 3)) for i in range(len(rows))]
    with get_context("fork").Pool(N_LAB_WORKERS, initializer=_score_init) as pool:
        lab = np.stack(pool.map(_score_one, tasks)).reshape(len(rows), 16, V, -1)
    cols = list(LC.CM.LABEL_COLS)
    tab = VV.variant_table(combine="cross")
    worst = {}
    for v, (a, d) in enumerate(tab):
        si = speeds.index(a)
        li = lats.index("none" if d == 0 else f"r{d:+.1f}")
        ref = lab_ref[:, :, si, li]
        mine = lab[:, :, v]
        assert np.array_equal(np.isnan(ref), np.isnan(mine))
        dv = np.nan_to_num(np.abs(mine - ref), nan=0.0).max(axis=(0, 1))
        worst[VV.variant_name(a, d)] = float(dv.max())
        for c in ("nc", "dac", "ttc", "comfort", "ddc"):
            assert np.array_equal(mine[..., cols.index(c)], ref[..., cols.index(c)]), (v, c)
        assert dv.max() <= 1e-5, (VV.variant_name(a, d), dict(zip(cols, dv.tolist())))
        assert np.array_equal(mine.astype(np.float32), ref.astype(np.float32), equal_nan=True), v   # cf.npz is f32
    pas = lab[..., [cols.index(c) for c in ("nc", "dac", "ttc", "comfort")]].min(-1) >= 1
    print(f"\n[official labels] {len(rows)} tokens x 16 cand x {V} variants: max|dev| {worst}; "
          f"pass rate per variant {np.round(pas.mean((0, 1)), 3).tolist()}")
