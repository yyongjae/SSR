"""CK2 teacher trainer tests (CPU).

Dataset (real data, skipped when absent): shapes / dtypes, label alignment with labels/<split>/raw256 and the variant
files, the fixed variant subset == sampler epoch-0 near+mid (token / GT / label alignment), determinism (re-open,
pickle, epoch change), no GT-derived slot order, model-input isolation (metadata perturbation leaves the forward
unchanged; slot permutation permutes it), real DET BEV, label prior.
Trainer (synthetic in-memory data, real surrogate GT of two navtrain tokens): loss forward / backward (every trainable
parameter gets a finite gradient, lon / gate heads frozen, z_lon == 0), metric functions on known answers, batch
sharding, GPU guard, end-to-end CPU training with evaluations / ckpt_best / resume, and 2-process gloo DDP == 1 process.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck import train_ck2 as T2  # noqa: E402
from tools.ck.data import ck2_dataset as D2  # noqa: E402
from tools.ck.data.ck_dataset import collate_ck  # noqa: E402

REAL = Path(Cn.CK_DATA)
VAL = "navtrain_val"
TRAIN_TOKENS = ("1aa44d46e4ab5bc7", "2570fbfdf1835706")
CKI = list(Cn.CK_LABEL_IDX)
PDMS = Cn.LABEL_COLS.index("pdms")


def _have_real(split=VAL):
    need = [REAL / "packed" / split / "tokens.parquet", REAL / "labels" / split / "raw256" / "labels.npy",
            REAL / "ck2" / "labels" / split / D2.VAR_NAME / "labels.npy"]
    return all(p.is_file() for p in need)


real = pytest.mark.skipif(not _have_real(), reason="real CK2 val data absent")


@pytest.fixture(scope="module")
def vds():
    return D2.CK2Dataset(VAL, bev="none", n_var=32, root=REAL)


@pytest.fixture(autouse=True)
def _threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _rows(ds, n, seed=0):
    return np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)


# ----------------------------------------------------------------------------------------------- dataset (real)
@real
def test_item_shapes_and_dtypes(vds):
    it = vds[0]
    K, V = vds.K, vds.n_var
    assert K == 32 and V == 32
    exp = {"cand": ((K, 8, 3), np.float32), "status": ((8,), np.float32), "gt_traj": ((8, 3), np.float32),
           "y": ((K, 5), np.float32), "y_ok": ((K,), np.bool_), "y_pdms": ((K,), np.float32),
           "anc_idx": ((K,), np.int64), "anc_group": ((K,), np.int8), "anc_pi": ((K,), np.float64),
           "anc_w": ((K,), np.float32), "var_traj": ((V, 8, 3), np.float32), "var_y": ((V, 5), np.float32),
           "var_ok": ((V,), np.bool_), "var_pdms": ((V,), np.float32), "var_anchor": ((V,), np.int64),
           "var_v": ((V,), np.int8), "var_col": ((V,), np.int64), "var_pi": ((V,), np.float32)}
    for k, (shape, dt) in exp.items():
        assert it[k].shape == shape and it[k].dtype == dt, (k, it[k].shape, it[k].dtype)
    assert "bev" not in it                                         # bev 'none'
    b = collate_ck([vds[0], vds[1]])
    assert b["cand"].shape == (2, K, 8, 3) and b["var_traj"].shape == (2, V, 8, 3) and len(b["tokens"]) == 2
    assert np.isclose(it["anc_w"].sum(), 256.0, atol=1e-3)
    assert (np.bincount(it["anc_group"], minlength=3) == [8, 8, 16]).all()


@real
def test_label_alignment_raw256_and_variants(vds):
    lab = np.load(REAL / "labels" / VAL / "raw256" / "labels.npy", mmap_mode="r")
    vd = D2.var_dir(VAL, root=REAL)
    vl = np.load(vd / "labels.npy", mmap_mode="r")
    vt = np.load(vd / "traj.npy", mmap_mode="r")
    ix = np.load(vd / "index.npz")
    anchors = AS.load_anchors()
    for i in _rows(vds, 40):
        it = vds[int(i)]
        r = it["row"]
        assert it["token"] == vds.tokens[r]
        L = np.asarray(lab[r], np.float32)
        assert np.array_equal(it["cand"], anchors[it["anc_idx"]])
        assert np.array_equal(it["y"], L[it["anc_idx"]][:, CKI])
        assert np.array_equal(it["y_pdms"], L[it["anc_idx"], PDMS])
        assert it["y_ok"].all()
        c = it["var_col"]
        assert len(np.unique(c)) == len(c)
        assert np.array_equal(it["var_traj"], np.asarray(vt[r, c]))
        assert np.array_equal(it["var_y"], np.asarray(vl[r, c])[:, CKI])
        assert np.array_equal(it["var_pdms"], np.asarray(vl[r, c, PDMS]))
        assert np.array_equal(it["var_v"], c % 6) and (it["var_v"] >= 1).all()
        assert np.array_equal(it["var_anchor"], ix["anchor_idx"][r, c // 6])
        assert ix["valid"][r].reshape(-1)[c].all() and it["var_ok"].all()
        # identity column of every chosen parent = the anchor bytes (variant files are anchor-aligned)
        k = c // 6
        assert np.array_equal(np.asarray(vt[r, k * 6]), anchors[it["var_anchor"]])


@real
def test_raw256_pass_mask_equals_official_ep_file(vds):
    """The sampler design only uses NC / DAC / TTC / C: raw256 (f32, used here) and pdm_score_256_officialEP (f16, used
    to build the variants) must give the same pass mask (EP differs by f16 rounding only)."""
    src = AS.LabelSource()
    lab = np.load(REAL / "labels" / VAL / "raw256" / "labels.npy", mmap_mode="r")
    for r in _rows(vds, 100, 1):
        a = src.get(str(vds.tokens[r]))
        b = np.asarray(lab[r], np.float32)[:, CKI].T
        assert np.array_equal(AS.pass_mask(a), AS.pass_mask(b))
        assert np.abs(a[2] - b[2]).max() < 1e-3


@pytest.mark.parametrize("split", [VAL, "navtrain_train"])
def test_variant_subset_is_sampler_epoch0_near_mid(split):
    if not _have_real(split):
        pytest.skip("real data absent")
    ds = D2.CK2Dataset(split, bev="none", n_var=32, root=REAL)
    ix = np.load(D2.var_dir(split, root=REAL) / "index.npz")
    for r in _rows(ds, 150, 2):
        r = int(ds.rows[r])
        lab = ds.raw_labels(r)
        s = ds.sampler.sample(str(ds.tokens[r]), np.asarray(ds._arr("gt_traj")[r]), lab[:, CKI].T, epoch=0,
                              shuffle=False)
        assert np.array_equal(s["idx"][:16], ix["anchor_idx"][r].astype(np.int64)), r
        # the shuffled epoch-0 draw is the same set, so every variant parent is in the epoch-0 anchors
        sh = ds.sample_anchors(r, epoch=0, lab256=lab)
        assert set(sh["idx"].tolist()) == set(s["idx"].tolist())


@real
def test_determinism_and_epochs(vds):
    i = 5
    a, b = vds[i], vds[i]
    c = pickle.loads(pickle.dumps(vds))[i]
    d = D2.CK2Dataset(VAL, bev="none", n_var=32, root=REAL)[i]
    for x in (b, c, d):
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(x[k])), k
    e1 = pickle.loads(pickle.dumps(vds))
    e1.set_epoch(1)
    x = e1[i]
    near = lambda it: set(it["anc_idx"][it["anc_group"] == 0].tolist())
    assert near(x) == near(a)                                       # the 8 GT-nearest never change
    assert set(x["anc_idx"].tolist()) != set(a["anc_idx"].tolist())
    assert not np.array_equal(x["var_col"], a["var_col"])
    # draws depend on (seed, epoch, token) only, not on the dataset row order / limit
    sub = D2.CK2Dataset(VAL, bev="none", n_var=32, root=REAL, rows=vds.rows[[i]])[0]
    assert np.array_equal(sub["anc_idx"], a["anc_idx"]) and np.array_equal(sub["var_col"], a["var_col"])
    # --seed changes the variant subset, --sampler-seed the anchor draw
    s1 = D2.CK2Dataset(VAL, bev="none", n_var=32, root=REAL, seed=1)[i]
    assert np.array_equal(s1["anc_idx"], a["anc_idx"]) and not np.array_equal(s1["var_col"], a["var_col"])


@real
def test_no_gt_derived_slot_order(vds):
    rows = _rows(vds, 300, 3)
    pos0, near_pos, corr, mono = [], [], [], 0
    for i in rows:
        it = vds[int(i)]
        d = AS.gt_distance(vds.sampler.anchors, it["gt_traj"])[it["anc_idx"]]
        pos0.append(int(np.argmin(d)))
        near_pos.extend(np.flatnonzero(it["anc_group"] == 0).tolist())
        ps = (it["y"][:, [0, 1, 3, 4]] >= 1).all(1).astype(float)
        if ps.std() > 0:
            corr.append(np.corrcoef(np.arange(len(ps)), ps)[0, 1])
        c = it["var_col"]
        mono += int((np.diff(c) > 0).all() or (np.diff(c) < 0).all())
    h = np.bincount(pos0, minlength=32) / len(pos0)
    assert h.max() < 0.12, h                                         # uniform would be 1/32
    assert abs(np.mean(near_pos) - 15.5) < 1.5
    assert abs(np.mean(corr)) < 0.05
    assert mono == 0


@real
def test_model_sees_only_model_inputs(vds):
    """forward() reads bev / cand / status / var_traj only: perturbing labels / sampler metadata / GT leaves it bitwise
    unchanged; a slot permutation permutes the per-candidate outputs (no slot-position dependence)."""
    from navsim.agents.para_ssr.ck.model import build_ck
    net = build_ck("T", seed=0, norm=(np.zeros(256, np.float32), np.ones(256, np.float32))).eval()
    T2.freeze_unused(net)
    items = [vds[0], vds[1]]
    rng = np.random.default_rng(0)
    for it in items:
        it["bev"] = rng.standard_normal((256, 50, 100)).astype(np.float16)
        it["bev_ok"] = True
        for k in ("cand", "y", "y_ok", "y_pdms", "anc_idx", "anc_group", "anc_pi", "anc_w"):
            it[k] = it[k][:6]
        for k in ("var_traj", "var_y", "var_ok", "var_pdms", "var_anchor", "var_v", "var_col", "var_pi"):
            it[k] = it[k][:5]
    b = collate_ck(items)
    dev = torch.device("cpu")
    with torch.no_grad():
        o1 = T2.forward(net, b, False, dev, True)
        b2 = dict(b)
        for k in ("y", "y_pdms", "anc_pi", "anc_w", "var_y", "var_pdms", "var_pi", "gt_traj"):
            b2[k] = torch.rand_like(b[k].float())
        for k in ("anc_idx", "anc_group", "var_anchor", "var_v", "var_col"):
            b2[k] = b[k].flip(-1)
        b2["y_ok"], b2["var_ok"] = ~b["y_ok"], ~b["var_ok"]
        o2 = T2.forward(net, b2, False, dev, True)
        for k in ("score_logit", "extra_score_logit", "w_lat", "z_lon"):
            assert torch.equal(o1[k], o2[k]), k
        p = torch.tensor([3, 0, 5, 1, 4, 2])
        b3 = dict(b)
        b3["cand"] = b["cand"][:, p]
        o3 = T2.forward(net, b3, False, dev, True)
        assert torch.allclose(o3["score_logit"], o1["score_logit"][:, p], atol=1e-5)
        assert torch.allclose(o3["w_lat"], o1["w_lat"][:, p], atol=1e-5)
        assert float(o1["z_lon"].abs().max()) == 0.0
    assert set(D2.MODEL_INPUT_KEYS) == {"bev", "cand", "status", "var_traj"}


@real
def test_det_bev_real():
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    ds = D2.CK2Dataset(VAL, bev="T", n_var=8, root=REAL, limit=3)
    if not TeacherCache.for_subset("navtrain").has(str(ds.tokens[ds.rows[0]])):
        pytest.skip("DET cache absent")
    it = ds[0]
    assert it["bev_ok"] and it["bev"].shape == (256, 50, 100) and it["bev"].dtype == np.float16
    with pytest.raises(ValueError):
        D2.CK2Dataset(VAL, bev="M", root=REAL)                     # no ReSMap cache for navtrain_val


@real
def test_label_prior(vds):
    p = D2.ck2_label_prior(vds, 200)
    assert p.shape == (5,) and ((p > 0) & (p < 1)).all()
    assert p[4] > p[2]                                              # comfort mostly 1, EP lower


# ----------------------------------------------------------------------------------------------- EP target (real)
def _old_item_labels(ds, i, epoch):
    """the pre-option (official) label arithmetic of CK2Dataset.labels_item / ck_dataset, re-implemented verbatim"""
    r = int(ds.rows[i])
    lab = np.array(ds._arr("raw_labels")[r], np.float32)
    s = ds.sampler.sample(str(ds.tokens[r]), np.asarray(ds._arr("gt_traj")[r], np.float32), lab[:, CKI].T,
                          epoch=epoch, shuffle=True)
    L = lab[s["idx"]]
    out = {"anc_idx": s["idx"], "y": np.ascontiguousarray(L[:, CKI]),
           "y_ok": np.asarray(ds._arr("raw_ok")[r, s["idx"]], bool) & np.isfinite(L[:, CKI]).all(1)}
    cols, okp, _ = ds.variant_cols(r, epoch)
    vl = np.asarray(ds._arr("var_labels")[r, cols], np.float32)
    out["var_y"] = np.ascontiguousarray(vl[:, CKI])
    out["var_ok"] = np.asarray(ds._arr("var_ok")[r, cols], bool) & okp & np.isfinite(vl[:, CKI]).all(1)
    return out, L, vl


@real
def test_ep_target_dataset_official_identical_decoupled_only_ep(vds):
    """--ep-target: 'official' (default) reproduces the old label arithmetic bit for bit; 'decoupled' changes only the
    EP column of y / var_y (= ep_target.decoupled_ep of the same 9-column rows), never the anchor draw, the variant
    draw, the masks or the official PDMS; the score prior moves only in EP."""
    from navsim.agents.para_ssr.ck.ep_target import decoupled_ep
    assert vds.ep_target == "official" and vds.describe()["ep_target"] == "official"
    dd = D2.CK2Dataset(VAL, bev="none", n_var=32, root=REAL, ep_target="decoupled")
    assert dd.describe()["ep_target"] == "decoupled"
    with pytest.raises(ValueError):
        D2.CK2Dataset(VAL, bev="none", n_var=0, root=REAL, ep_target="bogus")
    n_diff = 0
    for i in _rows(vds, 30, seed=3):
        for e in (0, 2):
            a, b = vds.labels_item(int(i), epoch=e), dd.labels_item(int(i), epoch=e)
            old, L, vl = _old_item_labels(vds, int(i), e)
            for k in ("anc_idx", "y", "y_ok", "var_y", "var_ok"):
                assert a[k].dtype == old[k].dtype and a[k].tobytes() == old[k].tobytes(), k     # official == old
            assert set(a) == set(b)
            for k in a:
                if k in ("y", "var_y"):
                    assert np.array_equal(np.delete(a[k], 2, -1), np.delete(b[k], 2, -1)), k
                elif isinstance(a[k], np.ndarray):
                    assert a[k].tobytes() == b[k].tobytes(), k                 # draws, masks, pdms untouched
                else:
                    assert a[k] == b[k], k
            assert np.array_equal(b["y"][:, 2], decoupled_ep(L)) and np.array_equal(b["var_y"][:, 2], decoupled_ep(vl))
            m1 = (L[:, Cn.LBL["nc"]] * L[:, Cn.LBL["dac"]] * L[:, Cn.LBL["ddc"]]) == 1
            assert np.abs(b["y"][m1, 2] - a["y"][m1, 2]).max(initial=0.0) <= 1e-6
            n_diff += int((b["y"][:, 2] != a["y"][:, 2]).sum())
    assert n_diff > 0
    pa, pb = D2.ck2_label_prior(vds, 100), D2.ck2_label_prior(dd, 100)
    assert np.array_equal(np.delete(pa, 2), np.delete(pb, 2)) and pb[2] > pa[2]


@real
def test_ep_target_r34_eval_set():
    """CKDataset (r34 top-16 'cand' eval labels): official == the old y; decoupled changes only EP; y_pdms official."""
    from navsim.agents.para_ssr.ck.ep_target import decoupled_ep
    from tools.ck.data.ck_dataset import CKDataset
    lab = np.load(REAL / "labels" / VAL / "cand" / "labels.npy", mmap_mode="r")
    a = CKDataset(VAL, bev="none", k=16, labels="cand", gt=False, limit=40)
    b = CKDataset(VAL, bev="none", k=16, labels="cand", gt=False, limit=40, ep_target="decoupled")
    assert a.ep_target == "official"
    for i in range(0, 40, 3):
        x, z = a[i], b[i]
        L = np.array(lab[x["row"], :16], np.float32)
        assert x["y"].tobytes() == L[:, CKI].tobytes() and x["y_pdms"].tobytes() == z["y_pdms"].tobytes()
        assert np.array_equal(np.delete(x["y"], 2, -1), np.delete(z["y"], 2, -1))
        assert np.array_equal(z["y"][:, 2], decoupled_ep(L)) and np.array_equal(x["y_ok"], z["y_ok"])


@real
def test_ep_target_build_datasets_and_config(tmp_path):
    """--ep-target reaches every dataset build_datasets makes (train, r34 val, raw val, train-eval sets)."""
    for ep in ("official", "decoupled"):
        a = T2.resolve_args(T2.get_parser().parse_args(
            ["--arm", "T", "--run", "x", "--device", "cpu", "--out-root", str(tmp_path), "--limit-tokens", "4",
             "--val-limit", "4", "--train-eval-rows", "4", "--eval-n-var", "4", "--n-var", "8", "--ep-target", ep]))
        d = T2.build_datasets(a)
        assert d["train"].ep_target == ep and len(d["evals"]) == 4
        assert all(ds.ep_target == ep for _, _, ds in d["evals"])
    a = T2.resolve_args(T2.get_parser().parse_args(["--arm", "M", "--run", "x"]))
    assert a.ep_target == "official"                                    # default: old behaviour
    with pytest.raises(SystemExit):
        T2.get_parser().parse_args(["--arm", "T", "--run", "x", "--ep-target", "bogus"])


# ----------------------------------------------------------------------------------------------- synthetic data
def _gt(token: str):
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    try:
        d = GTLoader(Cn.DATA_ROOT).load(token)
        if bool(d["ref_gt_ok"]):
            return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in d.items()}
    except Exception:
        pass
    g = GTLoader.__new__(GTLoader)
    from navsim.agents.para_ssr.refiner import data as RD
    g._A, g._CL, g._NKF, g._EH, g._EW = RD.A_MAX, RD.CL_MAX, RD.N_KF, RD.E_H, RD.E_W
    d = {f"ref_{k}": np.asarray(v) for k, v in g._empty().items()}
    d["ref_gt_ok"] = np.asarray(False)
    return d


_GT_CACHE = {}


def _gt_cached(token):
    if token not in _GT_CACHE:
        _GT_CACHE[token] = _gt(token)
    return _GT_CACHE[token]


def _paths(n, v0, rng, lat=0.0):
    t = np.arange(1, 9) * 0.5
    out = np.zeros((n, 8, 3), np.float32)
    for j in range(n):
        v = v0 * rng.uniform(0.5, 1.3)
        k = rng.normal(scale=0.02)
        x = v * t
        out[j, :, 0] = x
        out[j, :, 1] = lat + 0.5 * k * x ** 2
        out[j, :, 2] = np.arctan(k * x)
    return out


def _labels(rng, n):
    y = np.ones((n, 5), np.float32)
    f = rng.random(n) < 0.35
    key = rng.choice([0, 1, 3, 4], size=n)
    y[np.flatnonzero(f), key[f]] = 0.0
    y[:, 2] = rng.random(n).astype(np.float32)
    pd_ = y[:, 0] * y[:, 1] * (5 * y[:, 3] + 2 * y[:, 4] + 5 * y[:, 2]) / 12.0
    return y, pd_.astype(np.float32)


class FakeCK2(torch.utils.data.Dataset):
    """CK2Dataset-like items (K anchors + V variants), deterministic per (row, epoch); real surrogate GT."""

    def __init__(self, n=8, K=6, V=4, gt=True, seed=0, all_ok=True):
        self.n, self.K, self.n_var, self.gt, self.seed, self.all_ok = n, K, V, gt, seed, all_ok
        rng = np.random.default_rng(seed)
        self.bev = (rng.standard_normal((n, 256, 50, 100)) * 0.5).astype(np.float16)
        self.tokens = np.array([TRAIN_TOKENS[i % 2] for i in range(n)])
        self.rows = np.arange(n)
        self.epoch = 0
        self.var_meta = {"names": ["id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5"]}

    def set_epoch(self, e):
        self.epoch = int(e)

    def describe(self):
        return {"fake": True, "n": self.n}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        rng = np.random.default_rng([self.seed, r, self.epoch])
        K, V = self.K, self.n_var
        v0 = 4.0 + r
        cand = _paths(K, v0, rng)
        y, pdm = _labels(rng, K)
        vt = cand[:V].copy()
        vt[..., 1] += 0.3
        vy, vpd = _labels(rng, V)
        st = np.zeros(8, np.float32)
        st[1], st[4] = 1.0, v0
        anc = (np.arange(K) + 10 * r).astype(np.int64)
        it = dict(token=f"{self.tokens[r]}_{r}", row=r, bev=self.bev[r], bev_ok=True, cand=cand, status=st,
                  gt_traj=_paths(1, v0, np.random.default_rng(r))[0], y=y, y_pdms=pdm,
                  y_ok=np.ones(K, bool) if self.all_ok else rng.random(K) > 0.2, anc_idx=anc,
                  anc_group=np.array([0, 0, 1, 1, 2, 2] * K, np.int8)[:K], anc_pi=np.ones(K), anc_w=np.ones(K, np.float32),
                  var_traj=vt, var_y=vy, var_pdms=vpd, var_ok=np.ones(V, bool), var_anchor=anc[:V].copy(),
                  var_v=np.array([1, 2, 3, 4, 5] * V, np.int8)[:V], var_col=np.arange(V, dtype=np.int64),
                  var_pi=np.full(V, 0.5, np.float32))
        if self.gt:
            it.update(_gt_cached(str(self.tokens[r])))
        return it


class FakeR34(torch.utils.data.Dataset):
    """CKDataset(labels='cand')-like items (r34 eval path)."""

    def __init__(self, n=6, k=5, seed=1):
        rng = np.random.default_rng(seed)
        self.n, self.k = n, k
        self.bev = (rng.standard_normal((n, 256, 50, 100)) * 0.5).astype(np.float16)
        self.cand = np.stack([_paths(k, 5 + i, rng) for i in range(n)])
        lab = [_labels(rng, k) for _ in range(n)]
        self.y = np.stack([x[0] for x in lab])
        self.pd = np.stack([x[1] for x in lab])
        self.v2_final = rng.standard_normal((n, k)).astype(np.float32)
        self.v2_im = rng.dirichlet(np.ones(k), n).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        st = np.zeros(8, np.float32)
        st[1], st[4] = 1.0, 5.0 + i
        return dict(token=f"r34_{i}", row=i, bev=self.bev[i], bev_ok=True, cand=self.cand[i], status=st,
                    y=self.y[i], y_ok=np.ones(self.k, bool), y_pdms=self.pd[i], v2_final=self.v2_final[i],
                    v2_im=self.v2_im[i], v2_sim=self.y[i], gt_traj=self.cand[i, 0])


def _norm():
    return np.zeros(256, np.float32), np.ones(256, np.float32)


def _args(tmp_path, run="t", extra=()):
    a = T2.get_parser().parse_args(["--arm", "T", "--run", run, "--device", "cpu", "--out-root", str(tmp_path),
                                    "--workers", "0", "--tokens-per-batch", "2", "--epochs", "2", "--log-every", "1",
                                    "--ckpt-every", "0", "--train-eval-rows", "0", *extra])
    return T2.resolve_args(a)


# ----------------------------------------------------------------------------------------------- loss / metrics
def test_loss_forward_backward_all_trainable_params(tmp_path):
    from navsim.agents.para_ssr.ck.model import build_ck
    net = build_ck("T", seed=0, norm=_norm())
    frozen = T2.freeze_unused(net)
    assert frozen and all(n.startswith(("trunk.lon_head", "trunk.gate_head")) for n in frozen)
    ds = FakeCK2(n=2)
    b = collate_ck([ds[0], ds[1]])
    a = _args(tmp_path)
    dev = torch.device("cpu")
    out = T2.forward(net, b, False, dev, True)
    corr = T2.lateral_decode(b["cand"], out["w_lat"], out["ego"][0], Cn.LON_ST_SLOPE["train"])
    loss, st = T2.compute_loss(out, corr, b, a)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None, n
    assert st["z_lon_absmax"] == 0.0 and st["n_gt"] == 2.0 and "sur" in st and "t_cmf" in st and "t_prog" in st
    for k in Cn.CK_KEYS:
        lo, hi = sorted((st[f"anc_bce_{k}"], st[f"var_bce_{k}"]))
        assert lo - 1e-6 <= st[f"bce_{k}"] <= hi + 1e-6, k
    assert st["n_anc_ok"] == 12 and st["n_var_ok"] == 8
    # z_lon = 0 decode: no longitudinal change (identity in mode A with z = 0, w = 0 at init)
    assert torch.allclose(corr["traj"], b["cand"], atol=1e-5)


def test_sur_groups_masks_the_surrogate(tmp_path):
    """--sur-groups near / near_mid = correction_loss with the anc_group mask; all = unmasked (default)."""
    from navsim.agents.para_ssr.ck import losses as Lm
    from navsim.agents.para_ssr.ck.model import build_ck
    net = build_ck("T", seed=0, norm=_norm())
    T2.freeze_unused(net)
    with torch.no_grad():
        net.trunk.lat_head[-1].bias.fill_(0.3)                     # non-identity lateral correction
    b = collate_ck([FakeCK2(n=2)[0], FakeCK2(n=2)[1]])
    out = T2.forward(net, b, False, torch.device("cpu"), True)
    corr = T2.lateral_decode(b["cand"], out["w_lat"], out["ego"][0], Cn.LON_ST_SLOPE["train"])
    sur = {g: T2.compute_loss(out, corr, b, _args(tmp_path, extra=["--sur-groups", g]))[1]["sur"]
           for g in T2.SUR_GROUPS}
    assert _args(tmp_path).sur_groups == "all"
    idx = T2.TC.gt_index(b)
    n, K = int(idx.numel()), b["cand"].shape[1]
    sb = Lm.surrogate_batch_k({k: v for k, v in b.items() if k.startswith("ref_")}, idx, b["cand"].float(),
                              b["gt_traj"].float(), out["ego"][0], out["ego"][1])
    for g, valid in (("all", None), ("near", b["anc_group"][idx].long() <= 0),
                     ("near_mid", b["anc_group"][idx].long() <= 1)):
        ref, _, _ = Lm.correction_loss(corr["raw"], sb, n, K, Cn.SUR_WEIGHTS, Cn.SUR_MARGINS, valid=valid)
        assert sur[g] == pytest.approx(float(ref.detach()), rel=1e-6, abs=1e-8), g


def test_torchrun_passes_trainer_options():
    """torchrun's argparse (Python 3.9) scans the script's arguments too and aborts on an ambiguous abbreviation of
    its own options: '--run' (--run-path / --run_path) -> the --run-name spelling.  Every other option passes."""
    from torch.distributed.run import get_args_parser
    tp = get_args_parser()
    head = ["--standalone", "--nproc_per_node=2", "tools/ck/train_ck2.py"]
    for act in T2.get_parser()._actions:
        for o in act.option_strings:
            if o in ("-h", "--help", "--run"):
                continue
            assert tp.parse_args(head + ["--arm", "T", o, "1"]).training_script_args == ["--arm", "T", o, "1"], o
    ns = tp.parse_args(head + ["--arm", "T", "--run-name", "x", "--resume"])
    a = T2.get_parser().parse_args(ns.training_script_args)
    assert a.run == "x" and a.resume and a.arm == "T"


def test_bev_bad_masks_every_loss():
    ds = FakeCK2(n=2)
    items = [ds[0], ds[1]]
    items[1]["bev_ok"] = False
    b = T2.mask_bev_ok(collate_ck(items))
    assert not b["y_ok"][1].any() and not b["var_ok"][1].any() and not bool(b["ref_gt_ok"][1])
    assert b["y_ok"][0].all() and b["var_ok"][0].all()


def test_key_and_raw_metrics_known_answers():
    rng = np.random.default_rng(0)
    T, K, V = 40, 8, 6
    y = np.stack([_labels(rng, K)[0] for _ in range(T)])
    pdm = y[..., 0] * y[..., 1] * (5 * y[..., 3] + 2 * y[..., 4] + 5 * y[..., 2]) / 12
    vy = np.stack([_labels(rng, V)[0] for _ in range(T)])
    vpd = vy[..., 0] * vy[..., 1] * (5 * vy[..., 3] + 2 * vy[..., 4] + 5 * vy[..., 2]) / 12
    eps = 1e-4
    R = dict(prob=np.clip(y, eps, 1 - eps), y=y, y_ok=np.ones((T, K), bool), y_pdms=pdm,
             anc_group=np.tile(np.array([0, 0, 1, 1, 2, 2, 2, 2], np.int8), (T, 1)),
             anc_idx=np.tile(np.arange(K), (T, 1)), var_prob=np.clip(vy, eps, 1 - eps), var_y=vy,
             var_ok=np.ones((T, V), bool), var_pdms=vpd, var_v=np.tile(np.arange(1, V + 1) % 6, (T, 1)),
             var_anchor=np.tile(np.arange(V) % 4, (T, 1)), e_abs=np.zeros(3), n_tok=np.asarray(T),
             n_bev_bad=np.asarray(0))
    m = T2.eval_metrics("raw", R, ["id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5"])
    for k in ("nc", "dac", "ttc", "comfort"):
        assert m[f"anc_auc_fail_{k}"] == pytest.approx(1.0) and m[f"var_auc_fail_{k}"] == pytest.approx(1.0)
    assert m["anc_auc_pass"] == pytest.approx(1.0) and m["anc_ep_mae"] < 1e-3
    assert m["anc_pdms_sel"] == pytest.approx(m["anc_pdms_oracle"], abs=1e-6)
    assert m["pair_frac_parent_found"] == 1.0 and m["pair_n"] == T * V
    for k in ("nc", "dac", "ttc", "comfort", "ep", "pdms"):
        if m[f"pair_n_{k}"]:
            assert m[f"pair_acc_{k}"] == pytest.approx(1.0), k
    # inverted probabilities -> AUC 0, pair accuracy 0
    R2 = dict(R, prob=1 - R["prob"], var_prob=1 - R["var_prob"])
    m2 = T2.eval_metrics("raw", R2)
    assert m2["anc_auc_fail_dac"] == pytest.approx(0.0) and m2["pair_acc_dac"] == pytest.approx(0.0)
    # r34 kind
    v2f = rng.standard_normal((T, K)).astype(np.float32)
    r = T2.eval_metrics("r34", dict(prob=R["prob"], y=y, y_ok=R["y_ok"], y_pdms=pdm, v2_final=v2f,
                                    v2_im=np.full((T, K), 1.0 / K, np.float32), e_abs=np.zeros(2),
                                    n_tok=np.asarray(T), n_bev_bad=np.asarray(0)))
    assert r["pdms_ck_noim"] == pytest.approx(r["pdms_oracle"], abs=1e-6)
    assert r["auc_fail_comfort"] == pytest.approx(1.0) and r["n_tokens"] == T


def test_dist_epoch_batches_partition():
    n, B = 23, 4
    g = T2.TC.EpochBatches(n, B, 3, 2).batches()
    parts = [T2.DistEpochBatches(n, B, 3, 2, 0, r, 2).batches() for r in range(2)]
    assert len(parts[0]) == len(g) == n // B
    for j, b in enumerate(g):
        assert np.array_equal(np.concatenate([parts[0][j], parts[1][j]]), b)
    assert list(T2.DistEpochBatches(n, B, 3, 2, 2, 1, 2))[0] == [int(i) for i in parts[1][2]]
    assert len(T2.DistEpochBatches(n, B, 3, 2, 2, 1, 2)) == len(g) - 2
    with pytest.raises(ValueError):
        T2.DistEpochBatches(n, 3, 0, 0, 0, 0, 2)
    m = T2.merge_stats([{"loss": 1.0, "n_gt": 2.0}, {"loss": 3.0, "n_gt": 4.0, "skipped": 1.0}])
    assert m == {"loss": 2.0, "n_gt": 6.0, "skipped": 1.0}


def test_gpu_guard_ddp(monkeypatch):
    for vis, w, ok in (("0", 1, True), ("0,1", 2, True), ("2,3", 2, True), ("0,1", 1, False), ("0,4", 2, False),
                       ("0,1,2", 3, False), ("1,1", 2, False), ("", 1, False)):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", vis)
        if ok:
            U.gpu_guard_ddp("cuda", w)
        else:
            with pytest.raises(SystemExit):
                U.gpu_guard_ddp("cuda", w)
    U.gpu_guard_ddp("cpu", 2)


# ----------------------------------------------------------------------------------------------- end to end (CPU)
def _datasets(n_train=8, n_eval=4):
    return {"train": FakeCK2(n=n_train), "evals": [("navtrain_val", "r34", FakeR34(n=n_eval)),
                                                   ("navtrain_val_raw", "raw", FakeCK2(n=n_eval, gt=False, seed=5))]}


def test_train_cpu_end_to_end_and_resume(tmp_path):
    from navsim.agents.para_ssr.ck.model import build_ck, load_ck
    a = _args(tmp_path, "e2e", ["--max-steps", "3"])
    ds = _datasets()
    run = T2.train(a, datasets=ds, norm_override=_norm())
    assert (run / "ckpt_last.pt").is_file() and json.loads((run / "done.json").read_text())["steps"] == 3
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in log if r["kind"] == "step"]
    assert [r["step"] for r in steps] == [1, 2, 3]
    assert all(r["z_lon_absmax"] == 0.0 and r["world"] == 1 for r in steps)
    assert all(np.isfinite(r["loss"]) for r in steps)
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()]
    assert {(r["split"], r["kind"]) for r in val} == {("navtrain_val", "r34"), ("navtrain_val_raw", "raw")}
    vr = [r for r in val if r["kind"] == "raw"][0]
    for k in ("anc_auc_pass", "var_auc_pass", "pair_acc_pdms", "anc_pdms_sel", "var_bce_ep"):
        assert k in vr, k
    assert "pdms_a_b1" in [r for r in val if r["kind"] == "r34"][0]
    assert (run / "ckpt_best.pt").is_file() and json.loads((run / "best.json").read_text())["step"] == 3
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["trainer"] == "train_ck2" and cfg["world_size"] == 1 and cfg["global_batch"] == 2
    assert cfg["init_report"]["frozen"]
    # loaders of ck/model.py
    for which in ("last", "best"):
        net, c = load_ck(run, which)
        assert float(net.trunk.lon_head[-1].weight.detach().abs().max()) == 0.0
    net2 = build_ck("T", init_from=run)
    assert net2.init_report["kind"] == "ck"
    # resume to the end of 2 epochs (4 steps / epoch); done.json of the --max-steps stop removed first
    (run / "done.json").unlink()
    a2 = _args(tmp_path, "e2e", ["--resume"])
    T2.train(a2, datasets=ds, norm_override=_norm())
    done = json.loads((run / "done.json").read_text())
    assert done["steps"] == 8 and done["epochs_done"] == 2
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if r["kind"] == "step"] == list(range(1, 9))
    assert (run / "ckpt_ep0.pt").is_file() and (run / "ckpt_ep1.pt").is_file()
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()]
    assert sorted({r["step"] for r in val}) == [3, 4, 8]
    # finished run returns immediately; an existing ckpt without --resume refuses
    assert T2.train(_args(tmp_path, "e2e", ["--resume"]), datasets=ds, norm_override=_norm()) == run
    (run / "done.json").unlink()
    with pytest.raises(SystemExit):
        T2.train(_args(tmp_path, "e2e"), datasets=ds, norm_override=_norm())


def test_ep_target_config_and_resume_guard(tmp_path, capsys):
    """config.json records ep_target (+ rule); --resume refuses a different ep_target; a config.json written before
    the option (no key) counts as 'official'."""
    a = _args(tmp_path, "ep", ["--max-steps", "1", "--ep-target", "decoupled", "--best-metric", "none"])
    run = T2.train(a, datasets=_datasets(), norm_override=_norm())
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["ep_target"] == "decoupled" and "raw_progress" in cfg["ep_target_rule"]
    (run / "done.json").unlink()
    with pytest.raises(SystemExit, match="ep_target"):
        T2.train(_args(tmp_path, "ep", ["--resume", "--max-steps", "2", "--best-metric", "none"]),
                 datasets=_datasets(), norm_override=_norm())
    # an old run: no ep_target key -> official
    cfg.pop("ep_target")
    cfg.pop("ep_target_rule")
    U.write_json(run / "config.json", cfg)
    with pytest.raises(SystemExit, match="ep_target"):
        T2.train(_args(tmp_path, "ep", ["--resume", "--max-steps", "2", "--best-metric", "none",
                                        "--ep-target", "decoupled"]), datasets=_datasets(), norm_override=_norm())
    T2.train(_args(tmp_path, "ep", ["--resume", "--max-steps", "2", "--best-metric", "none"]),
             datasets=_datasets(), norm_override=_norm())
    assert json.loads((run / "done.json").read_text())["steps"] == 2
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()]
    assert "pdms_ck_plugin" in [r for r in val if r["kind"] == "r34"][0]   # descriptive plugin-weight selection


def test_eval_every_and_resume_config_guard(tmp_path):
    a = _args(tmp_path, "ev", ["--eval-every", "2", "--epochs", "3", "--best-metric", "none"])
    run = T2.train(a, datasets=_datasets(), norm_override=_norm())
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()]
    assert sorted({r["epoch"] for r in val}) == [1, 2]                 # epochs 1 (every 2) and 2 (last)
    assert not (run / "ckpt_best.pt").exists()
    (run / "done.json").unlink()
    with pytest.raises(SystemExit):
        T2.train(_args(tmp_path, "ev", ["--resume", "--n-var", "8"]), datasets=_datasets(), norm_override=_norm())


# ----------------------------------------------------------------------------------------------- DDP (gloo, CPU)
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _ddp_worker(rank, world, port, out_root, run, extra):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    a = T2.resolve_args(T2.get_parser().parse_args(
        ["--arm", "T", "--run", run, "--device", "cpu", "--out-root", out_root, "--workers", "0",
         "--tokens-per-batch", "4", "--epochs", "1", "--log-every", "1", "--ckpt-every", "0",
         "--train-eval-rows", "0", "--lr", "1e-4", "--warmup-steps", "1", "--clip", "0", *extra]))
    T2.train(a, datasets=_datasets(n_train=8, n_eval=5), norm_override=_norm())


def test_ddp_two_ranks_equals_single_process(tmp_path):
    import torch.multiprocessing as mp
    if torch.cuda.is_available():
        # fork-based ranks fail ("autograd's threading in combination with fork") once this process ran an autograd
        # backward with CUDA devices visible (earlier tests): rerun this test alone in a CPU-only interpreter
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                            str(tmp_path / "child"), f"{__file__}::test_ddp_two_ranks_equals_single_process"],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=CK, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        return
    from navsim.agents.para_ssr.ck.model import load_ck
    for t in TRAIN_TOKENS:
        _gt_cached(t)                                    # warm (forked children inherit it)
    if not all(bool(_GT_CACHE[t]["ref_gt_ok"]) for t in TRAIN_TOKENS):
        pytest.skip("surrogate GT of the fixture tokens absent")
    out = str(tmp_path)
    mp.start_processes(_ddp_worker, args=(2, _free_port(), out, "ddp2", ()), nprocs=2, join=True,
                       start_method="fork")
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    _ddp_worker_single(out)
    r2, r1 = Path(out) / "ddp2", Path(out) / "ddp1"
    assert (r2 / "done.json").is_file() and (r1 / "done.json").is_file()
    l2 = [json.loads(x) for x in (r2 / "train_log.jsonl").read_text().splitlines() if '"step"' in x]
    l1 = [json.loads(x) for x in (r1 / "train_log.jsonl").read_text().splitlines() if '"step"' in x]
    s2 = [r for r in l2 if r["kind"] == "step"]
    s1 = [r for r in l1 if r["kind"] == "step"]
    assert len(s2) == len(s1) == 2 and all(r["world"] == 2 for r in s2)
    assert json.loads((r2 / "done.json").read_text())["n_tok_last_session"] == 8
    for x, y in zip(s1, s2):
        assert x["loss"] == pytest.approx(y["loss"], rel=1e-4, abs=1e-6)
        assert x["n_gt"] == y["n_gt"] == 4.0 and x["n_anc_ok"] == y["n_anc_ok"]
        assert x["grad_norm"] == pytest.approx(y["grad_norm"], rel=1e-3)
    n2, _ = load_ck(r2, "last")
    n1, _ = load_ck(r1, "last")
    d = max(float((p - q).abs().max()) for p, q in zip(n2.state_dict().values(), n1.state_dict().values())
            if p.dtype.is_floating_point)
    assert d < 1e-5, d                                   # measured 8e-7 (reduction order only)
    v2 = [json.loads(x) for x in (r2 / "val_metrics.jsonl").read_text().splitlines()]
    v1 = [json.loads(x) for x in (r1 / "val_metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in v2] == [r["split"] for r in v1]
    for x, y in zip(v1, v2):
        assert x["n_tokens"] == y["n_tokens"] == 5
        for k in ("bce_dac", "anc_bce_dac", "var_bce_nc"):
            if k in x:
                assert x[k] == pytest.approx(y[k], rel=1e-3, abs=1e-5), k


def _ddp_worker_single(out):
    a = T2.resolve_args(T2.get_parser().parse_args(
        ["--arm", "T", "--run", "ddp1", "--device", "cpu", "--out-root", out, "--workers", "0",
         "--tokens-per-batch", "4", "--epochs", "1", "--log-every", "1", "--ckpt-every", "0",
         "--train-eval-rows", "0", "--lr", "1e-4", "--warmup-steps", "1", "--clip", "0"]))
    T2.train(a, datasets=_datasets(n_train=8, n_eval=5), norm_override=_norm())
