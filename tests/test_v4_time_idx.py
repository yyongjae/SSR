"""refiner v4: first-failure time-index sidecars + P_ref (navsim/agents/para_ssr/ck/time_idx.py, tools/ck/v4/build_time_idx.py,
the optional extras of tools/ck/data/ck2_dataset.py, ck/e2e_data2.py, ck/cands2.py, tools/ck/e2e2/labeler2.py).
CPU only; <= 2 worker processes; synthetic label dirs under the pytest tmp dir (real data read-only).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_time_idx.py
  CK_V4_REAL_TIDX=1 additionally builds partial sidecars from 2 real shards of raw256 / the merged accstraight
  variant dir (written to the tmp dir) and checks them against labels.npy, CK2Dataset and the official labeler2.

Off = bit-identical: the default code paths are compared with the git versions of the edited modules at CK_V4_BASE_REF
(default HEAD; once these edits are committed, pass the parent commit, e.g. CK_V4_BASE_REF=b43abf7, to keep the
comparison meaningful).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.ck.data import label_cands as LC  # noqa: E402  (pins navsim)
from tools.ck.data import common as CM  # noqa: E402
from tools.ck.v4 import build_time_idx as B  # noqa: E402
from tools.ck.e2e import labeler as L1  # noqa: E402
from tools.ck.e2e2 import labeler2 as L2  # noqa: E402
from navsim.agents.para_ssr.ck import cands2 as C2  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import time_idx as TI  # noqa: E402

SPLIT = "navtrain_train"
LBL = {c: i for i, c in enumerate(LC.LABEL_COLS)}
CKD = Path("/home/external-user/ssd/yongjae_refiner/ck")
PK = CKD / "packed" / SPLIT
RL = CKD / "labels" / SPLIT / "raw256"
VNAME = "var_separate_sampler16_accstraight"
VD = CKD / "ck2" / "labels" / SPLIT / VNAME
HAVE = all(p.is_file() for p in (PK / "tokens.parquet", RL / "labels.npy", VD / "labels.npy", VD / "index.npz"))
need = pytest.mark.skipif(not HAVE, reason="real packed / raw256 / variant label files absent")
BASE_REF = os.environ.get("CK_V4_BASE_REF", "HEAD")
REAL = pytest.mark.skipif(os.environ.get("CK_V4_REAL_TIDX", "") != "1" or not HAVE,
                          reason="real-shard sidecar build: set CK_V4_REAL_TIDX=1")
torch.set_num_threads(1)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint8)


def _eq(a, b):
    """bitwise equality of two values (numpy / torch / scalars), NaN == NaN by bits."""
    a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.numpy() if torch.is_tensor(b) else np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(_bits(a), _bits(b))


def _head_module(rel: str, name: str, tmp: Path):
    """the BASE_REF (git) version of a repo module, imported under `name` (package-relative imports resolve against the
    current package).  Skips when git is unavailable."""
    try:
        src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASE_REF}:{rel}"], text=True)
    except Exception as e:   # pragma: no cover
        pytest.skip(f"git show {BASE_REF}:{rel} failed: {e!r}")
    f = tmp / "head" / "a" / "b" / Path(rel).name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------------------------- fake label dirs
def _packed(root: Path, n: int, seed: int = 0):
    toks = [f"{seed:02d}tok{i:011d}" for i in range(n)]
    pk = CM.packed_dir(SPLIT, root)
    pk.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"token": toks, "log": ["L"] * n, "row": np.arange(n)}).to_parquet(pk / "tokens.parquet", index=False)
    return toks, pk


def _cell(rng, r, kk, crc, tok, pref):
    nc = float(rng.choice([0.0, 0.5, 1.0, 1.0]))
    dac = float(rng.choice([0.0, 1.0, 1.0]))
    ttc = float(rng.choice([0.0, 1.0]))
    ti = {"nc_time_idx": int(rng.integers(0, 41)) if nc < 1 else -1,
          "ttc_time_idx": int(rng.integers(0, 41)) if ttc < 1 else -1,
          "dac_time_idx": int(rng.integers(0, 41)) if dac < 1 else -1}
    return dict(token=tok, row=r, k=kk, crc=crc, nc=nc, dac=dac, ddc=1.0, ep=float(rng.random()), ttc=ttc,
                comfort=1.0, pdms=float(rng.random()), mult=nc * dac, raw_progress=float(10 * rng.random()),
                pdm_progress_eff=pref + 1e-7 * kk, nc_track="", nc_obj_type="", ttc_track="", **ti,
                ddc_stat_m=0.0, pdm_nc=1.0, pdm_dac=1.0, pdm_ddc=1.0, pdm_raw_progress=pref, rec_ok=True, error="",
                sec_load=0.01, sec_score=0.02)


def make_label_dir(d: Path, toks, traj: np.ndarray, chunk: int, seed: int, no_time_shards=(), err_rows=(),
                   notok=(), root: Path = None):
    """label_cands-format dir d: traj.npy, shards/c*.parquet (chunk rows each, every score_token column), then
    labels.npy / ok.npy / meta.json by label_cands.assemble (the real row / k mapping).  -> expected time_idx
    [n, K, 3] (-2 where not ok / no time columns / error row)."""
    rng = np.random.default_rng(seed)
    n, k = traj.shape[:2]
    sd = d / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    np.save(d / "traj.npy", traj)
    exp = TI.unknown((n, k))
    for cid in range(0, n, chunk):
        rows = []
        for r in range(cid, min(n, cid + chunk)):
            if r in err_rows:
                rows.append(dict(token=toks[r], row=r, k=-1, crc=-1, error="boom"))
                continue
            crc = CM.traj_crc(traj[r])
            for kk in range(k):
                c = _cell(rng, r, kk, crc, toks[r], 10.0 + r)
                if (r, kk) in notok:
                    c["pdms"] = float("nan")
                rows.append(c)
                if (r, kk) not in notok and cid // chunk not in no_time_shards:
                    exp[r, kk] = [c["nc_time_idx"], c["ttc_time_idx"], c["dac_time_idx"]]
        df = pd.DataFrame(rows)
        if cid // chunk in no_time_shards:
            df = df.drop(columns=list(TI.TIDX_SRC_COLS))
        CM.atomic_to_parquet(df, sd / f"c{cid // chunk:05d}.parquet")
    CM.atomic_write_json(sd / "_plan.json", {"token_sha16": CM.sha16_array(np.array(toks, "U16")), "chunk": chunk,
                                             "k": k, "rows": "", "n_rows": n})
    src = LC.TrajSource("npy", str(d / "traj.npy"), list(toks), k).open()
    LC.assemble(SPLIT, d.name, root, src, k, {}, out_dir=d)
    return exp


@pytest.fixture()
def fake(tmp_path):
    n, k = 9, 5
    root = tmp_path / "root"
    toks, pk = _packed(root, n)
    traj = np.random.default_rng(3).normal(size=(n, k, 8, 3)).astype(np.float32)
    d = root / "labels" / SPLIT / "fake5"
    exp = make_label_dir(d, toks, traj, chunk=4, seed=1, no_time_shards=(1,), err_rows=(2,), notok={(1, 3)}, root=root)
    return {"root": root, "toks": toks, "pk": pk, "dir": d, "exp": exp, "n": n, "k": k, "traj": traj}


def _run(f, out, **kw):
    kw.setdefault("workers", 2)
    return B.run(f["dir"], out, packed=str(f["pk"]), log=lambda s: None, **kw)


# ----------------------------------------------------------------------------------------------- builder
def test_builder_fake_shards_with_and_without_time_columns(fake, tmp_path):
    out = tmp_path / "out"
    meta = _run(fake, out)
    T, m2 = TI.load_time_idx(out, labels=fake["dir"] / "labels.npy", check_values=True)
    assert T.shape == (fake["n"], fake["k"], 3) and T.dtype == np.int8 and m2 == meta
    assert np.array_equal(np.asarray(T), fake["exp"])
    # shard 1 (rows 4..7) has no time columns -> -2; error-only row 2 -> -2; not-ok (1, 3) -> -2
    assert (T[4:8] == TI.UNKNOWN).all() and (T[2] == TI.UNKNOWN).all() and (T[1, 3] == TI.UNKNOWN).all()
    assert (T[0] != TI.UNKNOWN).all() and (T[8] != TI.UNKNOWN).all()
    ch = meta["checks"]
    assert ch["n_shards_no_time"] == 1 and ch["n_stale"] == 0 and ch["n_label_mismatch"] == 0
    assert ch["n_crc_checked"] == fake["n"] - 1 and ch["n_err_rows"] == 1 and ch["n_ok"] == ch["n_cells"] - 1
    assert meta["consistency"]["n_violations"] == 0 and meta["consistency"]["n_checked"] == int((T[..., 0] > -2).sum())
    assert not meta["partial"] and meta["rows_built"] == fake["n"] - 1 and meta["plan"]["equal"]
    lab = np.load(fake["dir"] / "labels.npy")
    assert meta["labels_sha16"] == CM.sha16_array(lab) == TI.sha16_array(lab)
    assert meta["counts"]["n_known_cells"] == int((fake["exp"][..., 0] != -2).sum())
    # consistency also holds against labels.npy itself (the array the sidecar is aligned with)
    ok = np.load(fake["dir"] / "ok.npy")
    assert TI.consistency(lab, T, ok & (np.asarray(T)[..., 0] > -2))["n_violations"] == 0
    # never overwritten without force
    with pytest.raises(SystemExit, match="exists"):
        _run(fake, out)
    assert _run(fake, out, force=True)["counts"] == meta["counts"]


def test_builder_consistency_violation_reported(fake, tmp_path):
    f = fake["dir"] / "shards" / "c00000.parquet"
    df = pd.read_parquet(f)
    i = int(np.flatnonzero((df.k >= 0) & (df.nc < 1) & df.pdms.notna())[0])
    df.loc[i, "nc_time_idx"] = -1                                    # NC failed but "no collision": inconsistent
    j = int(np.flatnonzero((df.k >= 0) & (df.dac == 1) & df.pdms.notna())[0])
    df.loc[j, "dac_time_idx"] = 99                                   # out of range -> -2 + counted
    df.to_parquet(f, index=False)
    meta = _run(fake, tmp_path / "out")
    c = meta["consistency"]
    assert c["violations"] == {"nc": 1, "ttc": 0, "dac": 0} and c["n_violations"] == 1
    assert c["examples"][0][:3] == [int(df.token.map({t: r for r, t in enumerate(fake["toks"])})[i]), int(df.k[i]),
                                    "nc"]
    assert meta["checks"]["n_bad_value"] == 1
    T = np.load(tmp_path / "out" / TI.FILE)
    assert T[int(df.row[j]), int(df.k[j]), 2] == TI.UNKNOWN


def test_builder_stale_and_label_guards(fake, tmp_path):
    traj = np.load(fake["dir"] / "traj.npy")
    t2 = traj.copy()
    t2[5, 0, 0, 0] += 1.0                                            # trajectories changed after scoring
    np.save(fake["dir"] / "traj.npy", t2)
    with pytest.raises(SystemExit, match="stale"):
        _run(fake, tmp_path / "o1")
    assert not (tmp_path / "o1" / TI.FILE).exists()
    assert _run(fake, tmp_path / "o1b", no_crc=True)["checks"]["crc_guard"].startswith("off")
    np.save(fake["dir"] / "traj.npy", traj)
    lab = np.load(fake["dir"] / "labels.npy")
    lab2 = lab.copy()
    lab2[6, 1, LBL["ep"]] += 0.5                                     # labels.npy no longer the shards' assembly
    np.save(fake["dir"] / "labels.npy", lab2)
    with pytest.raises(SystemExit, match="labels.npy"):
        _run(fake, tmp_path / "o2")
    np.save(fake["dir"] / "labels.npy", lab)
    ok = np.load(fake["dir"] / "ok.npy")
    ok2 = ok.copy()
    ok2[2, 0] = True                                                 # an ok cell no shard explains
    np.save(fake["dir"] / "ok.npy", ok2)
    with pytest.raises(SystemExit, match="covered by no shard"):
        _run(fake, tmp_path / "o3")
    np.save(fake["dir"] / "ok.npy", ok)
    assert _run(fake, tmp_path / "o4")["checks"]["n_stale"] == 0


def test_builder_partial_rows_and_max_shards(fake, tmp_path):
    meta = _run(fake, tmp_path / "p1", rows="0:2")
    T = np.load(tmp_path / "p1" / TI.FILE)
    assert meta["partial"] and meta["checks"]["n_shards_read"] == 1 and meta["rows_built"] == 2
    assert np.array_equal(T[:2], fake["exp"][:2]) and (T[2:] == TI.UNKNOWN).all()
    meta = _run(fake, tmp_path / "p2", max_shards=1)
    T = np.load(tmp_path / "p2" / TI.FILE)
    assert meta["partial"] and np.array_equal(T[:4], fake["exp"][:4]) and (T[4:] == TI.UNKNOWN).all()
    assert B.parse_rows("8", 9).tolist() == [8] and B.parse_rows("1,3,99", 9).tolist() == [1, 3]
    np.save(tmp_path / "r.npy", np.array([7, 0, 7]))
    assert B.parse_rows(str(tmp_path / "r.npy"), 9).tolist() == [0, 7] and B.parse_rows("", 9) is None


def test_builder_cli(fake, tmp_path):
    out = tmp_path / "cli"
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(REPO / "tools/ck/v4/build_time_idx.py"), "--label-dir", str(fake["dir"]),
                        "--packed", str(fake["pk"]), "--out", str(out), "--workers", "1"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert np.array_equal(np.load(out / TI.FILE), fake["exp"])
    r = subprocess.run([sys.executable, str(REPO / "tools/ck/v4/build_time_idx.py"), "--label-dir", str(fake["dir"]),
                        "--packed", str(fake["pk"]), "--out", str(out)], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode != 0 and "exists" in r.stderr


# ----------------------------------------------------------------------------------------------- merge accstraight
def make_merge(root: Path, n=6, K=2, seed=0):
    """fake base (16-anchor-style: K anchors x 6 VNAMES), alt (K x (id, a+0.5)) label dirs with shards and the merged
    dir (a+0.5 columns from alt) exactly as the real var_separate_sampler16_accstraight was made."""
    toks, pk = _packed(root, n, seed)
    rng = np.random.default_rng(seed)
    lab = root / "ck2" / "labels" / SPLIT
    names_b, names_a = list(D2.VNAMES), ["id", "a+0.5"]
    Vb, Va = len(names_b), len(names_a)
    tb = rng.normal(size=(n, K * Vb, 8, 3)).astype(np.float32)
    ta = rng.normal(size=(n, K * Va, 8, 3)).astype(np.float32)
    ta[:, ::Va] = tb[:, ::Vb]                                       # identity columns are the same anchors
    aidx = rng.integers(0, 256, (n, K)).astype(np.int16)
    vb = rng.random((n, K, Vb)) > 0.2
    va = np.ones((n, K, Va), bool)
    vb[:, :, 3] = True
    out = {}
    for name, traj, names, valid in (("var_separate_sampler16", tb, names_b, vb),
                                     ("var_accel_straight_sampler16", ta, names_a, va)):
        d = lab / name
        out[name] = make_label_dir(d, toks, traj, chunk=4, seed=seed + len(names), root=root)
        V = len(names)
        col = np.arange(K * V)
        np.savez(d / "index.npz", names=np.array(names), col_k=(col // V).astype(np.int16),
                 col_v=(col % V).astype(np.int16), anchor_idx=aidx, valid=valid, built=np.ones(n, bool))
    md = lab / VNAME
    md.mkdir(parents=True)
    a_cols = np.arange(K) * Vb + 3
    tm = tb.copy()
    tm[:, a_cols] = ta[:, np.arange(K) * Va + 1]
    Lb, La = np.load(lab / "var_separate_sampler16/labels.npy"), np.load(lab / "var_accel_straight_sampler16/labels.npy")
    Ob, Oa = np.load(lab / "var_separate_sampler16/ok.npy"), np.load(lab / "var_accel_straight_sampler16/ok.npy")
    Lm, Om = Lb.copy(), Ob.copy()
    Lm[:, a_cols], Om[:, a_cols] = La[:, np.arange(K) * Va + 1], Oa[:, np.arange(K) * Va + 1]
    np.save(md / "traj.npy", tm)
    np.save(md / "labels.npy", Lm)
    np.save(md / "ok.npy", Om)
    col = np.arange(K * Vb)
    np.savez(md / "index.npz", names=np.array(names_b), col_k=(col // Vb).astype(np.int16),
             col_v=(col % Vb).astype(np.int16), anchor_idx=aidx, valid=vb, built=np.ones(n, bool))
    rec = {"from": f"{SPLIT}/var_separate_sampler16", "a+0.5_from": f"{SPLIT}/var_accel_straight_sampler16"}
    (md / "meta.json").write_text(json.dumps({"split": SPLIT, "n": n, "k": K * Vb, "merged": rec}))
    (md / "build.json").write_text(json.dumps({"names": names_b, "merged": {"a+0.5_from": "var_accel_straight_sampler16",
                                                                           "ext_a+0.5": "straight"}}))
    exp = out["var_separate_sampler16"].copy()
    exp[:, a_cols] = out["var_accel_straight_sampler16"][:, np.arange(K) * Va + 1]
    return {"md": md, "pk": pk, "exp": exp, "a_cols": a_cols, "K": K, "toks": toks}


def test_merge_accstraight_mapping(tmp_path):
    m = make_merge(tmp_path / "root")
    meta = B.run_merge(m["md"], out=tmp_path / "out", packed=str(m["pk"]), workers=2, log=lambda s: None)
    T, _ = TI.load_time_idx(tmp_path / "out", labels=m["md"] / "labels.npy")
    assert np.array_equal(np.asarray(T), m["exp"])
    mg = meta["merge"]
    assert mg["alt_names"] == ["a+0.5"] and mg["n_alt_cols"] == m["K"] and mg["verify"]["n_rows"] == 6
    assert all(mg["verify"][k] == 0 for k in ("n_traj_mismatch", "n_crc_mismatch", "n_label_mismatch_rows",
                                              "n_ok_mismatch_rows"))
    assert meta["consistency"]["n_violations"] == 0 and meta["mode"] == "merge_accstraight" and not meta["partial"]
    assert meta["consistency"]["n_checked"] == int((np.asarray(T)[..., 0] > -2).sum()) > 0
    assert set(meta["consistency_per_source"]) == {"base", "alt"}
    cm = B.column_map(B._index(m["md"]), B._index(m["md"].parent / "var_separate_sampler16"),
                      B._index(m["md"].parent / "var_accel_straight_sampler16"), ["a+0.5"])
    assert cm["alt_cols"].tolist() == m["a_cols"].tolist() and cm["src_col"][m["a_cols"]].tolist() == [1, 3]


@pytest.mark.parametrize("what", ["traj_swap", "labels", "anchor_idx", "valid", "names"])
def test_merge_accstraight_refuses_mismatch(tmp_path, what):
    m = make_merge(tmp_path / "root")
    md = m["md"]
    if what == "traj_swap":                     # merged a+0.5 column of row 1 = the BASE a+0.5 (wrong source)
        t = np.load(md / "traj.npy")
        t[1, m["a_cols"][0]] = np.load(md.parent / "var_separate_sampler16/traj.npy")[1, m["a_cols"][0]]
        np.save(md / "traj.npy", t)
    elif what == "labels":
        L = np.load(md / "labels.npy")
        L[3, 0, LBL["ttc"]] = 0.25
        np.save(md / "labels.npy", L)
    else:
        z = dict(np.load(md / "index.npz"))
        if what == "anchor_idx":
            z["anchor_idx"] = z["anchor_idx"].copy()
            z["anchor_idx"][2, 1] += 1
        elif what == "valid":
            z["valid"] = ~z["valid"]
        else:
            z["names"] = np.array(["id", "a-1.0", "a-0.5", "a+0.7", "l-0.5", "l+0.5"])
        np.savez(md / "index.npz", **z)
    with pytest.raises(SystemExit):
        B.run_merge(md, out=tmp_path / "out", packed=str(m["pk"]), workers=1, log=lambda s: None)
    assert not (tmp_path / "out" / TI.FILE).exists()


# ----------------------------------------------------------------------------------------------- reader
def write_sidecar(d: Path, T: np.ndarray, labels_path: Path, partial: bool = False) -> Path:
    """a minimal valid sidecar (tests): time_idx.npy + meta.json with the labels fingerprints."""
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / TI.FILE, np.ascontiguousarray(T, np.int8))
    L = np.load(labels_path, mmap_mode="r")
    (d / TI.META).write_text(json.dumps({"version": TI.VERSION, "n": int(T.shape[0]), "k": int(T.shape[1]),
                                         "cols": list(TI.TIDX_COLS), "partial": partial,
                                         "labels_sha16": TI.sha16_array(L), "labels_probe_sha16": TI.probe_sha16(L)}))
    return d


def consistent_tidx(lab9: np.ndarray, ok: np.ndarray, seed: int = 0) -> np.ndarray:
    """random time indices consistent with labels [..., 9] (label < 1 <=> idx >= 0) on ok cells, -2 elsewhere."""
    rng = np.random.default_rng(seed)
    T = np.empty(lab9.shape[:-1] + (3,), np.int64)
    for j, li in enumerate(TI.TIDX_LABEL_IDX):
        T[..., j] = np.where(lab9[..., li] < 1, rng.integers(0, 41, lab9.shape[:-1]), -1)
    T[~ok] = TI.UNKNOWN
    return T.astype(np.int8)


def test_reader_checks_and_helpers(fake, tmp_path):
    out = tmp_path / "out"
    meta = _run(fake, out)
    a, m = TI.load_time_idx(out, expect_sha16_of_labels=meta["labels_sha16"], labels=np.load(fake["dir"] / "labels.npy"))
    assert isinstance(a, np.memmap) and m["n"] == fake["n"]
    with pytest.raises(TI.TimeIdxError, match="sha16"):
        TI.load_time_idx(out, expect_sha16_of_labels="0" * 16)
    L = np.load(fake["dir"] / "labels.npy")
    L2 = L.copy()
    L2[0, 0, 0] = 0.123                                              # a different label file (row 0 is probed)
    with pytest.raises(TI.TimeIdxError, match="probe"):
        TI.load_time_idx(out, labels=L2)
    with pytest.raises(TI.TimeIdxError):
        TI.load_time_idx(out, labels=L[:, :3])
    np.save(tmp_path / "bad.npy", np.zeros((fake["n"], fake["k"] + 1, 3), np.int8))
    shutil.copy(tmp_path / "bad.npy", out / TI.FILE)
    with pytest.raises(TI.TimeIdxError, match="int8"):
        TI.load_time_idx(out)
    with pytest.raises(FileNotFoundError):
        TI.load_time_idx(tmp_path / "nothing")
    # helpers
    t = np.array([[-2, -1, 0], [5, 40, 41]])
    assert TI.sanitize(t).tolist() == [[-2, -1, 0], [5, 40, -2]] and TI.sanitize(t, [True, False])[1].tolist() == [-2] * 3
    s = TI.to_seconds(np.array([-2, -1, 0, 7, 40], np.int8))
    assert np.isnan(s[0]) and np.isinf(s[1]) and np.allclose(s[2:], [0.0, 0.7, 4.0]) and s.dtype == np.float32
    st = TI.to_seconds(torch.tensor([-2, -1, 7], dtype=torch.int8), none_value=9.0)
    assert torch.isnan(st[0]) and float(st[1]) == 9.0 and abs(float(st[2]) - 0.7) < 1e-6
    lab = np.random.default_rng(0).random((2, 3, 4, 9)).astype(np.float32)
    lab[0, 0, :2, 8] = np.nan
    lab[1, 2, :, 8] = np.nan
    p = TI.pref_of(lab)
    assert p.shape == (2, 3) and p[0, 0] == lab[0, 0, 2, 8] and np.isnan(p[1, 2]) and p[1, 0] == lab[1, 0, 0, 8]
    assert torch.equal(torch.nan_to_num(TI.pref_of(torch.from_numpy(lab)), -1.0), torch.from_numpy(np.nan_to_num(p, nan=-1.0)))
    assert TI.TIDX_LABEL_IDX == (LBL["nc"], LBL["ttc"], LBL["dac"]) and TI.PREF_IDX == LBL["pdm_progress_eff"] == 8


# ----------------------------------------------------------------------------------------------- CK2Dataset
def _small_ck2_root(tmp: Path, rows=(0, 17, 40000)):
    """3 real navtrain_train rows copied into a small CK data root (packed, raw256, the accstraight variant dir)."""
    root = tmp / "ckroot"
    pk, rl, vd = CM.packed_dir(SPLIT, root), CM.labels_dir(SPLIT, "raw256", root), root / "ck2/labels" / SPLIT / VNAME
    for d in (pk, rl, vd):
        d.mkdir(parents=True)
    rows = list(rows)
    tdf = pd.read_parquet(PK / "tokens.parquet")
    sub = tdf.iloc[rows].reset_index(drop=True)
    sub["row"] = np.arange(len(rows))
    sub.to_parquet(pk / "tokens.parquet", index=False)
    for f in ("gt_traj.npy", "status.npy", "ok.npy"):
        np.save(pk / f, np.asarray(np.load(PK / f, mmap_mode="r")[rows]))
    for f in ("labels.npy", "ok.npy"):
        np.save(rl / f, np.asarray(np.load(RL / f, mmap_mode="r")[rows]))
        np.save(vd / f, np.asarray(np.load(VD / f, mmap_mode="r")[rows]))
    np.save(vd / "traj.npy", np.asarray(np.load(VD / "traj.npy", mmap_mode="r")[rows]))
    z = np.load(VD / "index.npz")
    np.savez(vd / "index.npz", **{k: (z[k][rows] if z[k].shape[:1] == (D2.N_ROWS,) else z[k]) for k in z.files})
    shutil.copy(VD / "build.json", vd / "build.json")
    return root, vd


@need
def test_ck2_dataset_default_bit_identical_and_sidecar_keys(tmp_path):
    from tools.ck.data import ck2_dataset as CD
    H = _head_module("tools/ck/data/ck2_dataset.py", "tools.ck.data._head_ck2_dataset", tmp_path)
    root, vd = _small_ck2_root(tmp_path)
    kw = dict(bev="none", n_var=32, var_name=VNAME, gt=False, seed=0, sampler_seed=0, root=root)
    new, old = CD.CK2Dataset(SPLIT, **kw), H.CK2Dataset(SPLIT, **kw)
    assert len(new) == len(old) == 3 and new.describe() == old.describe()
    for i in range(3):
        for e in (0, 2):
            a, b = new.labels_item(i, epoch=e), old.labels_item(i, epoch=e)
            assert set(a) == set(b), set(a) ^ set(b)
            for k in a:
                assert (a[k] == b[k]) if isinstance(a[k], (str, int)) else _eq(a[k], b[k]), (i, e, k)
    assert np.allclose(CD.ck2_label_prior(new, 10), H.ck2_label_prior(old, 10))
    # sidecars (synthetic but consistent with the real labels) -> extra keys only
    tr = tmp_path / "tidx_root"
    rlab = CM.labels_dir(SPLIT, "raw256", root) / "labels.npy"
    Lr, Or = np.load(rlab), np.load(rlab.parent / "ok.npy")
    Lv, Ov = np.load(vd / "labels.npy"), np.load(vd / "ok.npy")
    Tr, Tv = consistent_tidx(Lr, Or, 1), consistent_tidx(Lv, Ov, 2)
    write_sidecar(tr / SPLIT / "raw256", Tr, rlab)
    write_sidecar(tr / SPLIT / VNAME, Tv, vd / "labels.npy")
    ds = CD.CK2Dataset(SPLIT, time_idx_root=tr, with_pref=True, **kw)
    assert set(ds.describe()) - set(new.describe()) == {"time_idx", "with_pref"}
    for i in range(3):
        a, b = ds.labels_item(i, epoch=1), new.labels_item(i, epoch=1)
        assert set(a) - set(b) == {"anc_tidx", "var_tidx", "pref"} and set(b) <= set(a)
        for k in b:
            assert (a[k] == b[k]) if isinstance(a[k], (str, int)) else _eq(a[k], b[k]), k
        r = a["row"]
        assert a["anc_tidx"].dtype == np.int8 and a["anc_tidx"].shape == (32, 3) and a["var_tidx"].shape == (32, 3)
        assert np.array_equal(a["anc_tidx"], np.where(a["y_ok"][:, None], Tr[r, a["anc_idx"]], -2))
        assert np.array_equal(a["var_tidx"], np.where(a["var_ok"][:, None], Tv[r, a["var_col"]], -2))
        # official labels: nc / ttc / dac target < 1 <=> index >= 0 on the ok candidates
        for t, y, okm in ((a["anc_tidx"], a["y"], a["y_ok"]), (a["var_tidx"], a["var_y"], a["var_ok"])):
            for j, key in enumerate(("nc", "ttc", "dac")):
                ci = ("nc", "dac", "ep", "ttc", "comfort").index(key)
                assert np.array_equal((y[okm, ci] < 1), (t[okm, j] >= 0)), key
        fin = np.flatnonzero(np.isfinite(Lr[r, :, 8]))
        assert a["pref"].dtype == np.float32 and a["pref"] == Lr[r, fin[0], 8]
    from tools.ck.data.ck_dataset import collate_ck
    bt = collate_ck([ds[i] for i in range(3)])
    assert bt["anc_tidx"].dtype == torch.int8 and bt["anc_tidx"].shape == (3, 32, 3) and bt["pref"].shape == (3,)
    # guards: a sidecar of other labels, a partial (test) sidecar
    write_sidecar(tr / SPLIT / VNAME, Tv, vd / "labels.npy", partial=True)
    with pytest.raises(TI.TimeIdxError, match="partial"):
        CD.CK2Dataset(SPLIT, time_idx_root=tr, **kw)
    assert CD.CK2Dataset(SPLIT, time_idx_root=tr, time_idx_partial_ok=True, **kw).tidx_meta["var_tidx"]["partial"]
    Lx = Lr.copy()
    Lx[0, 0, 1] = 0.5
    np.save(tmp_path / "other_labels.npy", Lx)
    write_sidecar(tr / SPLIT / "raw256", Tr, tmp_path / "other_labels.npy")
    with pytest.raises(TI.TimeIdxError, match="probe"):
        CD.CK2Dataset(SPLIT, time_idx_root=tr, time_idx_partial_ok=True, **kw)
    with pytest.raises(FileNotFoundError):
        CD.CK2Dataset(SPLIT, time_idx_root=tmp_path / "none", **kw)


# ----------------------------------------------------------------------------------------------- e2e warm-up
def _wu_small(tmp: Path, n: int = 4):
    """structurally valid n-row warm-up sources (labels from real raw256 / variant rows when present, else random) +
    consistent sidecars."""
    pk, rl, vd = tmp / "pk", tmp / "rl", tmp / "vd"
    for d in (pk, rl, vd):
        d.mkdir(parents=True)
    rng = np.random.default_rng(5)
    toks = [f"t{i:015d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": ["L"] * n, "row": np.arange(n)}).to_parquet(pk / "tokens.parquet")
    np.save(pk / "ok.npy", np.ones(n, bool))
    if HAVE:
        rows = [0, 17, 40000, 85000][:n]
        gt = np.asarray(np.load(PK / "gt_traj.npy", mmap_mode="r")[rows], np.float32)
        raw = np.asarray(np.load(RL / "labels.npy", mmap_mode="r")[rows], np.float32)
        vl = np.asarray(np.load(VD / "labels.npy", mmap_mode="r")[rows], np.float32)
        vt = np.asarray(np.load(VD / "traj.npy", mmap_mode="r")[rows], np.float32)
        z = np.load(VD / "index.npz")
        valid, aidx = z["valid"][rows], z["anchor_idx"][rows]
    else:   # pragma: no cover
        gt = np.cumsum(np.ones((n, 8, 3), np.float32), 1)
        raw = rng.random((n, 256, 9)).astype(np.float32)
        vl = rng.random((n, 96, 9)).astype(np.float32)
        vt = rng.normal(size=(n, 96, 8, 3)).astype(np.float32)
        valid, aidx = np.ones((n, 16, 6), bool), np.zeros((n, 16), np.int16)
    raw[1, :, 8] = np.nan                                         # row 1: no P_ref in raw256 -> variant row's
    vl[3, :, 8] = np.nan
    raw[3, :, 8] = np.nan                                         # row 3: none anywhere -> NaN
    np.save(pk / "gt_traj.npy", gt)
    np.save(rl / "labels.npy", raw)
    rok = np.ones((n, 256), bool)
    rok[3, 7] = False                                             # row 3 not a keep row (raw ok not all True)
    np.save(rl / "ok.npy", rok)
    np.save(vd / "traj.npy", vt)
    np.save(vd / "labels.npy", vl)
    vok = np.ones((n, 96), bool)
    vok[0, 13] = False
    np.save(vd / "ok.npy", vok)
    np.savez(vd / "index.npz", valid=valid, anchor_idx=aidx, built=np.ones(n, bool))
    Tr, Tv = consistent_tidx(raw, np.ones((n, 256), bool), 3), consistent_tidx(vl, np.ones((n, 96), bool), 4)
    write_sidecar(tmp / "t_raw", Tr, rl / "labels.npy")
    write_sidecar(tmp / "t_var", Tv, vd / "labels.npy")
    base = {"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)}
    return base, toks, raw, vl, Tr, Tv, rok, vok


def test_warmup_cfg_and_rows(tmp_path):
    base, toks, raw, vl, Tr, Tv, rok, vok = _wu_small(tmp_path)
    w0 = D2.warmup_cfg(base)
    assert not set(w0) & {"raw_tidx", "var_tidx", "pref"}               # absent -> the old dict
    w = D2.warmup_cfg({**base, "raw_tidx": str(tmp_path / "t_raw"), "var_tidx": "none", "pref": "true"})
    assert w["raw_tidx"] == str(tmp_path / "t_raw") and w["var_tidx"] is None and w["pref"] is True
    s0 = D2._WUArrays(w0)
    assert not s0.tidx and not s0.pref and "raw_tidx" not in s0.paths
    for r in (-1, 0, 2, 99):
        assert set(D2.wu_row(s0, r)) == set(D2.WU_TARGET_KEYS)
    src = D2._WUArrays(D2.warmup_cfg({**base, "raw_tidx": str(tmp_path / "t_raw"), "var_tidx": str(tmp_path / "t_var"),
                                      "pref": True}))
    assert src.tidx and src.pref and set(src.check_tidx()) == {"raw_tidx", "var_tidx"}
    for r in range(4):
        a, b = D2.wu_row(src, r), D2.wu_row(s0, r)
        assert set(a) - set(b) == {"ck2_raw_tidx", "ck2_var_tidx", "ck2_pref"}
        for k in b:
            assert _eq(a[k], b[k]), k
        assert np.array_equal(a["ck2_raw_tidx"], np.where(a["ck2_raw_ok"][:, None], Tr[r], -2))
        assert np.array_equal(a["ck2_var_tidx"], np.where(a["ck2_var_ok"][:, None], Tv[r], -2))
        exp = TI.pref_of(raw[r]) if np.isfinite(raw[r, :, 8]).any() else TI.pref_of(vl[r])
        assert _eq(a["ck2_pref"], np.float32(exp))
    assert a["ck2_raw_tidx"].dtype == np.int8 and np.isnan(D2.wu_row(src, 3)["ck2_pref"])
    assert D2.wu_row(src, 3)["ck2_raw_tidx"][7].tolist() == [-2, -2, -2]            # raw ok False -> unknown
    assert D2.wu_row(src, 0)["ck2_var_tidx"][13].tolist() == [-2, -2, -2]           # variant ok False -> unknown
    assert not D2.wu_row(src, 3)["ck2_wu_ok"] and (D2.wu_row(src, 3)["ck2_raw_tidx"][:7] >= -1).all()
    assert np.isfinite(D2.wu_row(src, 1)["ck2_pref"]) == bool(np.isfinite(vl[1, :, 8]).any())
    for r in (-1, 99):
        e = D2.wu_row(src, r)
        assert not e["ck2_wu_ok"] and (e["ck2_raw_tidx"] == -2).all() and (e["ck2_var_tidx"] == -2).all()
        assert np.isnan(e["ck2_pref"]) and set(e) == set(D2.wu_row(src, 0))
    # only the raw sidecar: the variant indices are all unknown (both keys still emitted)
    s1 = D2._WUArrays(D2.warmup_cfg({**base, "raw_tidx": str(tmp_path / "t_raw")}))
    o = D2.wu_row(s1, 2)
    assert (o["ck2_var_tidx"] == -2).all() and "ck2_pref" not in o and (o["ck2_raw_tidx"] >= -1).any()
    # a sidecar built for other labels is a configuration error
    write_sidecar(tmp_path / "t_bad", Tr, Path(base["var_dir"]) / "labels.npy")
    with pytest.raises(D2.WUSourceError):
        D2._WUArrays(D2.warmup_cfg({**base, "raw_tidx": str(tmp_path / "t_bad")})).check_tidx()


def _targets(src, rows):
    t = [{k: torch.from_numpy(np.array(v, copy=True)) for k, v in D2.wu_row(src, r).items()} for r in rows]
    return {k: torch.stack([x[k] for x in t]) for k in t[0]}


def test_warmup_cands_and_fill_fallback_extras(tmp_path):
    base, toks, raw, vl, Tr, Tv, rok, vok = _wu_small(tmp_path)
    s0 = D2._WUArrays(D2.warmup_cfg(base))
    s1 = D2._WUArrays(D2.warmup_cfg({**base, "raw_tidx": str(tmp_path / "t_raw"), "var_tidx": str(tmp_path / "t_var"),
                                     "pref": True}))
    rows = [0, 1, 2, -1]
    tk = [toks[0], toks[1], toks[2], "ffffffffffffffff"]
    t0, t1 = _targets(s0, rows), _targets(s1, rows)
    ws = D2.WarmupSampler(n_var=32)
    W0, W1 = C2.warmup_cands(t0, tk, 2, ws), C2.warmup_cands(t1, tk, 2, ws)
    assert set(W1) - set(W0) == {"tidx", "pref"} and "tidx" not in W0
    for k in W0:
        assert torch.equal(W0[k], W1[k]), k
    ai, vc, ok = W1["anc_idx"].numpy(), W1["var_cols"].numpy(), W1["ok"].numpy()
    for b in range(4):
        tr, tv = t1["ck2_raw_tidx"][b].numpy(), t1["ck2_var_tidx"][b].numpy()
        exp = np.concatenate([tr[ai[b]], tv[vc[b]]])
        assert np.array_equal(W1["tidx"][b].numpy(), np.where(ok[b][:, None], exp, -2))
    assert W1["tidx"].dtype == torch.int8 and W1["tidx"].shape == (4, 64, 3) and (W1["tidx"][3] == -2).all()
    assert torch.equal(torch.nan_to_num(W1["pref"], -7.0), torch.nan_to_num(t1["ck2_pref"].float(), -7.0))
    # fill_fallback: LabelStore2 without a generation -> every row a fallback row
    io = tmp_path / "io"
    for wt, wp in ((False, False), (True, True)):
        st = D2.LabelStore2(io, n_rows=4, packed=base["packed"], with_tidx=wt, with_pref=wp)
        st.refresh(4)
        G = st.lookup(torch.tensor(rows), tk, epoch=5, seed=0)
        assert ("G_tidx" in G) == wt and ("G_pref" in G) == wp and not G["has"].any()
        if wt:
            assert (G["G_tidx"] == -2).all() and G["G_tidx"].dtype == torch.int8 and torch.isnan(G["G_pref"]).all()
        G0 = C2.fill_fallback({k: v.clone() for k, v in G.items()}, t0, tk, 5, ws)     # targets without extras
        G1 = C2.fill_fallback({k: v.clone() for k, v in G.items()}, t1, tk, 5, ws)
        for k in ("G_traj", "G_y", "G_ok", "G_vtype", "G_col", "G_src", "G_epoch", "has"):
            assert torch.equal(G0[k], G1[k]), k
        if not wt:
            assert set(G1) == set(G0) == {"G_traj", "G_y", "G_ok", "G_vtype", "G_col", "G_src", "G_epoch", "has"}
            continue
        assert G1["G_src"].tolist() == [D2.SRC_FALLBACK] * 3 + [D2.SRC_NODATA]
        assert (G0["G_tidx"] == -2).all() and torch.isnan(G0["G_pref"]).all()
        S = ws.sample(tk, {k: t1[k].numpy() for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_var_valid")}, 5,
                      stream=D2.STREAM_LAB)
        for b in range(3):
            a16 = S["anc_idx"][b][S["group"][b] <= 1]
            exp = np.concatenate([t1["ck2_raw_tidx"][b].numpy()[a16], t1["ck2_var_tidx"][b].numpy()[S["var_cols"][b]]])
            assert np.array_equal(G1["G_tidx"][b].numpy(), np.where(G1["G_ok"][b].numpy()[:, None], exp, -2))
            assert _eq(G1["G_pref"][b], t1["ck2_pref"][b])
        assert (G1["G_tidx"][3] == -2).all() and torch.isnan(G1["G_pref"][3])
        Gs = C2.onpolicy_label_set(st, t1, torch.tensor(rows), tk, 5, 0, ws)
        assert torch.equal(Gs["G_tidx"], G1["G_tidx"])


@need
def test_e2e_default_paths_equal_head(tmp_path):
    """off = bit-identical: wu_row (real default sources), warm-up candidates, LabelStore2.lookup + fill_fallback and
    open_generation2 of the current modules == the HEAD versions."""
    HD = _head_module("navsim/agents/para_ssr/ck/e2e_data2.py", "navsim.agents.para_ssr.ck._head_e2e_data2", tmp_path)
    HC = _head_module("navsim/agents/para_ssr/ck/cands2.py", "navsim.agents.para_ssr.ck._head_cands2", tmp_path)
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rows = [0, 40000, len(tdf) - 1, -1]
    tk = [str(tdf.token.iloc[r]) if r >= 0 else "ffffffffffffffff" for r in rows]
    a_src, b_src = D2._WUArrays(D2.warmup_cfg(None)), HD._WUArrays(HD.warmup_cfg(None))
    assert D2.warmup_cfg(None) == HD.warmup_cfg(None)
    for r in rows:
        a, b = D2.wu_row(a_src, r), HD.wu_row(b_src, r)
        assert set(a) == set(b)
        for k in a:
            assert _eq(a[k], b[k]), (r, k)
    ta, tb = _targets(a_src, rows), _targets(b_src, rows)
    ws = D2.WarmupSampler(n_var=32)
    Wa, Wb = C2.warmup_cands(ta, tk, 3, ws), HC.warmup_cands(tb, tk, 3, ws)
    assert set(Wa) == set(Wb) and all(torch.equal(Wa[k], Wb[k]) for k in Wa)
    io = tmp_path / "io"
    n = 6
    rng = np.random.default_rng(0)
    gen = D2.open_generation2(io, 4, n, "w+")
    assert "time_idx" not in gen and not (ED.gen_dir(io, 4) / "time_idx.npy").exists()
    assert "optional_files" not in json.loads((ED.gen_dir(io, 4) / "meta.json").read_text())
    D2.write_generation_rows2(gen, [0, 2], rng.normal(size=(2, 96, 8, 3)), rng.random((2, 96, 9)),
                              np.ones((2, 96), bool), rng.random((2, 96)) > 0.2)
    sa, sb = D2.LabelStore2(io, n_rows=n), HD.LabelStore2(io, n_rows=n)
    assert sa.refresh(4)["n_prev"] == sb.refresh(4)["n_prev"] == 2
    r6 = torch.tensor([0, 1, 2, -1])
    t6 = [f"x{i}" for i in range(4)]
    Ga, Gb = sa.lookup(r6, t6, 5, 0), sb.lookup(r6, t6, 5, 0)
    assert set(Ga) == set(Gb) and all(torch.equal(Ga[k], Gb[k]) for k in Ga)
    tt = {k: v[[0, 1, 2, 3]] for k, v in ta.items()}
    Fa, Fb = C2.fill_fallback(Ga, tt, t6, 5, ws), HC.fill_fallback(Gb, tt, t6, 5, ws)
    assert set(Fa) == set(Fb) and all(torch.equal(Fa[k], Fb[k]) for k in Fa)


# ----------------------------------------------------------------------------------------------- generations / labeler2
N_FAKE = 12


def _tokens(tmp: Path, n=N_FAKE):
    df = pd.DataFrame(dict(token=[f"tok{i:04d}" for i in range(n)], log=["__fake__"] * n, row=np.arange(n)))
    p = tmp / "tokens.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def _chunk(io, epoch, rows, seed=0, attempt="aaaa0000", seq=0):
    g = np.random.default_rng(seed)
    rows = np.asarray(rows, np.int64)
    traj = g.normal(size=(len(rows), 96, 8, 3)).astype(np.float32)
    valid = g.random((len(rows), 96)) > 0.1
    valid[:, ::6] = True
    p = ED.rec_chunk_path(io, epoch, 0, attempt, seq)
    D2.write_rec2_chunk(p, rows, traj, valid, np.zeros((len(rows), 16)), np.ones((len(rows), 16)),
                        np.full(len(rows), 3.0), np.arange(len(rows)), epoch, 0)
    return traj, valid


def _lab(io, tok, **kw):
    return L2.Labeler2(io, 1, workers=2, poll=0.2, once=True, scorer="fake", n_rows=N_FAKE, status_every=0,
                       tokens=tok, log=lambda s: None, **kw)


def test_score_one2_fake_equals_score_one():
    traj = np.random.default_rng(0).normal(size=(96, 8, 3)).astype(np.float32)
    traj[4, 2, 1] = np.nan
    task = (("k",), "tok0001", traj, "fake:tok0001")
    old = L1._W.get("mode")
    L1._W["mode"] = "fake"
    try:
        a, b = L2.score_one2(task), L1.score_one(task)
    finally:
        if old is None:
            L1._W.pop("mode", None)
        else:
            L1._W["mode"] = old
    assert len(a) == 7 and a[0] == b[0] and _eq(a[1], b[1]) and _eq(a[2], b[2]) and a[3] == b[3] == ""
    t = a[6]
    assert t.dtype == np.int8 and t.shape == (96, 3) and (t[4] == -2).all() and (t[a[2]] >= -1).all()
    assert TI.consistency(a[1], t, a[2])["n_violations"] == 0 and TI.consistency(a[1], t, a[2])["n_checked"] == 95
    assert np.array_equal(t[a[2]], L2.fake_time_idx(traj[a[2]], a[1][a[2]]))


def test_labeler2_writes_time_idx_and_old_generation(tmp_path):
    tok = _tokens(tmp_path)
    a, b = tmp_path / "a" / "ck_e2e2", tmp_path / "b" / "ck_e2e2"
    for io in (a, b):
        tr, va = _chunk(io, 5, [0, 2, 4, 6])
    tr[1, 10, 0, 0] = np.nan                                       # an unscorable trajectory (row 2, column 10)
    for io in (a, b):
        D2.write_rec2_chunk(ED.rec_chunk_path(io, 5, 0, "aaaa0000", 0), [0, 2, 4, 6], tr, va, np.zeros((4, 16)),
                            np.ones((4, 16)), np.full(4, 3.0), np.arange(4), 5, 0)
    assert _lab(a, tok).run() == 0                                  # default: time_idx on
    assert _lab(b, tok, time_idx=False).run() == 0                  # old behaviour
    ga, gb = D2.open_generation2(a, 5, N_FAKE, "r", optional=("time_idx",)), D2.open_generation2(b, 5, N_FAKE, "r")
    for k in D2.GEN2_KEYS:
        assert _eq(np.asarray(ga[k]), np.asarray(gb[k])), k             # existing files bit-identical
    assert "time_idx" in ga and not (ED.gen_dir(b, 5) / "time_idx.npy").exists()
    ma = json.loads((ED.gen_dir(a, 5) / "meta.json").read_text())
    assert ma["optional_files"]["time_idx"][:3] == [[96, 3], "int8", -2] and "optional_files" not in \
        json.loads((ED.gen_dir(b, 5) / "meta.json").read_text())
    T = np.asarray(ga["time_idx"])
    for i, r in enumerate([0, 2, 4, 6]):
        ok = np.asarray(ga["cand_ok"][r])
        assert np.array_equal(T[r][ok], L2.fake_time_idx(tr[i][ok], np.asarray(ga["labels"][r])[ok]))
        assert (T[r][~ok] == -2).all()
        assert TI.consistency(np.asarray(ga["labels"][r]), T[r], ok)["n_violations"] == 0
    assert (T[2, 10] == -2).all() and not ga["cand_ok"][2, 10] and (T[[1, 3, 5]] == -2).all()
    # LabelStore2 on the OLD generation (no file): G_tidx -2 everywhere, G_* unchanged
    sb_t, sb = D2.LabelStore2(b, n_rows=N_FAKE, packed=str(tmp_path), with_tidx=True), \
        D2.LabelStore2(b, n_rows=N_FAKE, packed=str(tmp_path))
    sb_t.refresh(5), sb.refresh(5)
    toks = [f"tok{i:04d}" for i in range(N_FAKE)]
    rows = torch.tensor([0, 2, 3])
    Gt, G = sb_t.lookup(rows, [toks[0], toks[2], toks[3]], 6, 0), sb.lookup(rows, [toks[0], toks[2], toks[3]], 6, 0)
    assert (Gt["G_tidx"] == -2).all() and Gt["has"].tolist() == [True, True, False]
    assert all(torch.equal(Gt[k], G[k]) for k in G) and "G_pref" not in Gt
    # a new labeler writing new rows into the old generation adds the file: old rows -2, new rows real
    _chunk(b, 5, [1, 3], seed=7, attempt="bbbb0000")
    lb = _lab(b, tok)
    assert lb.run() == 0
    gb2 = D2.open_generation2(b, 5, N_FAKE, "r", optional=("time_idx",))
    T2 = np.asarray(gb2["time_idx"])
    assert (T2[[0, 2, 4, 6]] == -2).all() and (T2[[1, 3]] >= -1).any() and gb2["row_state"][[1, 3]].tolist() == [1, 1]
    assert "time_idx" in json.loads((ED.gen_dir(b, 5) / "meta.json").read_text())["optional_files"]
    # status / skip paths never add the file to a generation without new rows
    _chunk(tmp_path / "c", 4, [0], seed=1)
    assert _lab(tmp_path / "c", tok, time_idx=False).run() == 0
    assert _lab(tmp_path / "c", tok).run() == 0                     # nothing new to score
    assert not (ED.gen_dir(tmp_path / "c", 4) / "time_idx.npy").exists()


def test_labeler2_skip_invalid_time_idx(tmp_path):
    tok = _tokens(tmp_path)
    io = tmp_path / "ck_e2e2"
    tr, va = _chunk(io, 6, [3])
    assert (~va[0]).any()
    assert _lab(io, tok, skip_invalid=True).run() == 0
    g = D2.open_generation2(io, 6, N_FAKE, "r", optional=("time_idx",))
    assert (np.asarray(g["time_idx"][3])[~va[0]] == -2).all() and (np.asarray(g["time_idx"][3])[va[0]] >= -1).all()


def test_generation2_optional_files_and_store_gather(tmp_path):
    io = tmp_path / "io"
    n = 8
    rng = np.random.default_rng(2)
    gen = D2.open_generation2(io, 4, n, "w+", optional=("time_idx",))
    assert (np.asarray(gen["time_idx"]) == -2).all() and gen["time_idx"].dtype == np.int8
    with pytest.raises(ValueError):
        D2.open_generation2(io, 4, n, "r", optional=("nope",))
    rows = [1, 4, 6]
    tr = rng.normal(size=(3, 96, 8, 3)).astype(np.float32)
    lab = rng.random((3, 96, 9)).astype(np.float32)
    lab[1, :7, 8] = np.nan                                       # row 4: P_ref from column 7
    lab[2, :, 8] = np.nan                                        # row 6: no P_ref
    cok = rng.random((3, 96)) > 0.1
    va = rng.random((3, 96)) > 0.2
    va[:, ::6] = True
    T = consistent_tidx(lab, cok, 9)
    D2.write_generation_rows2(gen, rows, tr, lab, cok, va, time_idx=T)
    D2.write_generation_rows2(gen, [7], tr[:1], lab[:1], cok[:1], va[:1])        # no indices given -> -2
    assert (np.asarray(gen["time_idx"][7]) == -2).all()
    g_plain = D2.open_generation2(io, 4, n, "r")
    assert "time_idx" not in g_plain
    with pytest.raises(ValueError, match="optional"):
        D2.write_generation_rows2(g_plain, [0], tr[:1], lab[:1], cok[:1], va[:1], time_idx=T[:1])
    toks = [f"tok{i:04d}" for i in range(n)]
    st = D2.LabelStore2(io, n_rows=n, packed=str(tmp_path), with_tidx=True, with_pref=True)
    st0 = D2.LabelStore2(io, n_rows=n, packed=str(tmp_path))
    st.refresh(4), st0.refresh(4)
    q = torch.tensor([1, 4, 6, 0, -1])
    tq = [toks[1], toks[4], toks[6], toks[0], ""]
    G, G0 = st.lookup(q, tq, 5, 0), st0.lookup(q, tq, 5, 0)
    assert set(G) - set(G0) == {"G_tidx", "G_pref"} and all(torch.equal(G[k], G0[k]) for k in G0)
    assert G["G_tidx"].shape == (5, 48, 3) and G["G_pref"].dtype == torch.float32
    for b in range(3):
        c = G["G_col"][b].numpy().astype(np.int64)
        assert np.array_equal(G["G_tidx"][b].numpy(), np.where(G["G_ok"][b].numpy()[:, None], T[b][c], -2))
    assert float(G["G_pref"][0]) == lab[0, 0, 8] and float(G["G_pref"][1]) == lab[1, 7, 8]
    assert torch.isnan(G["G_pref"][2:]).all() and (G["G_tidx"][3:] == -2).all()
    # ok candidates: label < 1 <=> index >= 0 (the gathered G_y are the official CK targets)
    okm = G["G_ok"].numpy()
    for j, ci in zip(range(3), (0, 3, 1)):
        assert np.array_equal(G["G_y"].numpy()[..., ci][okm] < 1, G["G_tidx"].numpy()[..., j][okm] >= 0)
    import pickle
    st2 = pickle.loads(pickle.dumps(st))
    assert torch.equal(st2.lookup(q, tq, 5, 0)["G_tidx"], G["G_tidx"])


# ----------------------------------------------------------------------------------------------- real data (gated)
@REAL
def test_real_sidecars_from_two_shards(tmp_path):
    """2 real shards each: raw256 sidecar (label equality + crc guard + 0 consistency violations), the merged
    accstraight sidecar (mapping verified), CK2Dataset on them, and the official labeler2 on row 17's recorded 96
    variant trajectories == the merged sidecar row (same scorer, same trajectories)."""
    out = tmp_path / "tidx"
    m1 = B.run(RL, out / SPLIT / "raw256", max_shards=2, workers=2, log=lambda s: None)
    m2 = B.run_merge(VD, out=out / SPLIT / VNAME, max_shards=2, workers=2, log=lambda s: None)
    for m in (m1, m2):
        assert m["consistency"]["n_violations"] == 0 and m["consistency"]["n_checked"] > 0 and m["partial"]
        assert m["rows_built"] == 128
    assert m1["checks"]["n_label_mismatch"] == 0 and m1["checks"]["n_crc_checked"] == 128
    assert m2["merge"]["verify"]["n_rows"] == 128
    T, _ = TI.load_time_idx(out / SPLIT / "raw256", labels=RL / "labels.npy")
    L = np.asarray(np.load(RL / "labels.npy", mmap_mode="r")[:128])
    ok = np.asarray(np.load(RL / "ok.npy", mmap_mode="r")[:128])
    t = np.asarray(T[:128])
    assert (t[ok] >= -1).all() and TI.consistency(L, t, ok)["n_violations"] == 0
    from tools.ck.data import ck2_dataset as CD
    ds = CD.CK2Dataset(SPLIT, bev="none", n_var=32, var_name=VNAME, rows=np.arange(5), time_idx_root=out,
                       with_pref=True, time_idx_partial_ok=True)
    for i in range(5):
        it = ds.labels_item(i)
        okm = it["y_ok"]
        assert np.array_equal(it["y"][okm, 0] < 1, it["anc_tidx"][okm, 0] >= 0)
        assert np.array_equal(it["var_y"][it["var_ok"], 1] < 1, it["var_tidx"][it["var_ok"], 2] >= 0)
        assert np.isfinite(it["pref"])
    # official labeler2 (score_one2) on one real token == offline sidecar
    tt = pd.read_parquet(PK / "tokens.parquet")
    r = 17
    pd.DataFrame(dict(token=[tt.token.iloc[r]], log=[tt.log.iloc[r]], row=[0])).to_parquet(tmp_path / "t.parquet")
    io = tmp_path / "ck_e2e2"
    tr = np.asarray(np.load(VD / "traj.npy", mmap_mode="r")[[r]], np.float32)
    va = np.load(VD / "index.npz")["valid"][[r]].reshape(1, 96)
    D2.write_rec2_chunk(ED.rec_chunk_path(io, 5, 0, "a1b2c3d4", 0), [0], tr, va, np.zeros((1, 16)), np.zeros((1, 16)),
                        np.zeros(1), 0, 5, 0)
    lab = L2.Labeler2(io, 1, workers=1, poll=0.2, once=True, n_rows=1, tokens=str(tmp_path / "t.parquet"),
                      scorer="official", status_every=0, log=lambda s: None)
    assert lab.run() == 0
    g = D2.open_generation2(io, 5, 1, "r", optional=("time_idx",))
    Tm, _ = TI.load_time_idx(out / SPLIT / VNAME)
    assert np.array_equal(np.asarray(g["time_idx"][0]), np.asarray(Tm[r]))
    assert np.array_equal(_bits(np.asarray(g["labels"][0])), _bits(np.asarray(np.load(VD / "labels.npy",
                                                                                      mmap_mode="r")[r])))
