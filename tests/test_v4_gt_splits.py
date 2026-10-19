"""refiner v4_3 GT data for the held-out splits: tools/ck/v4/build_sdf_split.py (official-DAC SDF of navtrain_val /
navtest) and refiner.e2e.GTLoader(split=...) (ref_obj_* from CK/v3/gt_objects/<split>, ref_sdf from sdf/<split>).
CPU only; <= 2 worker processes; builds go to the pytest tmp dir (real data read-only).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_gt_splits.py

Train = bit-identical: GTLoader(split='navtrain_train') (and the default) is compared with the git version of e2e.py at
CK_V4_BASE_REF (default HEAD; once these edits are committed pass the parent commit) on real train tokens, with the
teacher BEVs and the human path.  Real-data tests skip when their inputs are absent; the sdf/<split> tests skip until
the split's SDF build is complete (manifest.json complete).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.ck.v4 import build_sdf_split as B  # noqa: E402
from navsim.agents.para_ssr.ck import constants as C  # noqa: E402
from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import e2e as E  # noqa: E402
from navsim.agents.para_ssr.refiner import sdf as SDF  # noqa: E402

DATA = Path(C.DATA_ROOT)
BASE_REF = os.environ.get("CK_V4_BASE_REF", "3b4bf71")    # pre-v4_3 GTLoader (pinned)
GTO = DATA / "ck/v3/gt_objects"
TRAIN_TOK = DATA / "splits/e2e_train_trainlogs.parquet"
NAVTRAIN_STATS = SDF.SDF_ROOT / "navtrain/_build/stats.jsonl"
TDET, TMAP = C.CK_DATA / "ck2/train/ck2T10", C.CK_DATA / "ck2/train/ck2M10"
HAVE_TRAIN = TRAIN_TOK.is_file() and (DATA / "objects/navtrain").is_dir() and (SDF.SDF_ROOT / "navtrain").is_dir()
HAVE_VAL_MC = Path(C.SPLITS["navtrain_val"]["tokens"]).is_file() and Path(C.SPLITS["navtrain_val"]["metric_cache"]).is_dir()
need_train = pytest.mark.skipif(not HAVE_TRAIN, reason="real navtrain GT stores absent")
need_val_mc = pytest.mark.skipif(not HAVE_VAL_MC, reason="navtrain_val tokens / metric cache absent")
torch.set_num_threads(1)


def _complete(split: str) -> bool:
    m = SDF.SDF_ROOT / split / "_build" / "manifest.json"
    try:
        return bool(json.loads(m.read_text())["complete"])
    except Exception:
        return False


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """dtype, shape and bytes equal (NaN == NaN)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    x, y = np.ascontiguousarray(a.numpy()), np.ascontiguousarray(b.numpy())
    return x.tobytes() == y.tobytes()


def _head_module(rel: str, name: str, tmp: Path):
    """the BASE_REF (git) version of a repo module, imported under `name` (package-relative imports resolve against the
    current package).  Skips when git is unavailable."""
    try:
        src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASE_REF}:{rel}"], text=True)
    except Exception as e:   # pragma: no cover
        pytest.skip(f"git show {BASE_REF}:{rel} failed: {e!r}")
    f = tmp / "head" / Path(rel).name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _split_sample(split: str, n: int, seed: int = 0):
    df = pd.read_parquet(C.SPLITS[split]["tokens"])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), n, replace=False))
    return df.iloc[idx][["token", "log"]].astype(str).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- GTLoader
@need_train
def test_gtloader_train_bit_identical(tmp_path):
    """split='navtrain_train' (and the default) == the git GTLoader: same keys, dtypes, shapes, bytes; human path and
    teacher BEVs included when the teacher runs exist; a token with no GT takes the same empty path."""
    head = _head_module("navsim/agents/para_ssr/refiner/e2e.py", "navsim.agents.para_ssr.refiner._e2e_head_gt", tmp_path)
    runs = tuple(str(r) for r in (TDET, TMAP) if (r / "config.json").is_file())
    toks = pd.read_parquet(TRAIN_TOK).token.astype(str).tolist()
    sample = [toks[i] for i in (0, 1, 1000, 30000, 60000, len(toks) - 1)] + ["ffffffffffffffff"]
    old = head.GTLoader(DATA, runs, human_path=True)
    new = [E.GTLoader(DATA, runs, human_path=True), E.GTLoader(DATA, runs, human_path=True, split="navtrain_train")]
    assert all(g.teacher_subset == "navtrain" and not g.part_flags for g in new)
    assert [str(p) for p in new[0].src.objects] == [str(p) for p in old.src.objects]
    assert str(new[0].src.sdf) == str(old.src.sdf)
    n_ok = 0
    for t in sample:
        a = old.load(t)
        for g in new:
            b = g.load(t)
            assert list(a) == list(b), (t, set(a) ^ set(b))
            for k in a:
                assert _same(a[k], b[k]), (t, k)
        n_ok += bool(a["ref_gt_ok"])
    assert n_ok == len(sample) - 1          # every real train token has full surrogate GT, the fake one none


def test_gtloader_bad_split():
    with pytest.raises(ValueError):
        E.GTLoader(DATA, split="navtrain")
    assert set(E.GT_SOURCES) == {"navtrain_train", "navtrain_val", "navtest"}


@pytest.mark.parametrize("split", ["navtrain_val", "navtest"])
def test_gtloader_heldout_sources(split):
    """ref_obj_* == the CK/v3 GT-object file through data._read_objects; ref_sdf == sdf/<split> (when built); part
    flags; ref_gt_ok False (no e2e_side for these splits); same ref_* keys / shapes / dtypes as train; teacher subset."""
    if not (GTO / split).is_dir() or not Path(C.SPLITS[split]["tokens"]).is_file():
        pytest.skip(f"{GTO / split} absent")
    g = E.GTLoader(DATA, split=split)
    assert g.part_flags and g.teacher_subset == C.SPLITS[split]["teacher_subset"]
    assert [str(p) for p in g.src.objects] == [str(GTO / split)] and g.src.sdf == DATA / "sdf" / split
    ref_keys = E.GTLoader(DATA).load("ffffffffffffffff")
    sdf_built = _complete(split)
    sample = _split_sample(split, 6, seed=1)
    n_obj = 0
    for t in sample.token:
        d = g.load(t)
        assert list(d) == list(ref_keys) + list(E.GT_PART_KEYS)
        for k, v in ref_keys.items():
            assert d[k].dtype == v.dtype and d[k].shape == v.shape, k
        o = RD._read_objects(SimpleNamespace(objects=[GTO / split]), t)
        assert o is not None and bool(d["ref_obj_ok"])
        for k in ("obj_kf", "obj_first", "obj_meta", "obj_n", "obj_n_kf", "obj_R", "obj_ego_kf"):
            assert _same(d[f"ref_{k}"], torch.as_tensor(np.asarray(o[k]))), (t, k)
        n_obj += int(d["ref_obj_n"])
        assert not bool(d["ref_side_ok"]) and not bool(d["ref_gt_ok"])
        if sdf_built:
            assert bool(d["ref_sdf_ok"])
            ref = SDF.load_sdf(DATA / "sdf" / split / f"{t}.npz")
            assert d["ref_sdf"].numpy().tobytes() == ref.tobytes()
    assert n_obj > 0
    e = g.load("ffffffffffffffff")                         # absent token: empty GT, every flag False
    assert not any(bool(e[k]) for k in E.GT_PART_KEYS + ("ref_gt_ok",)) and int(e["ref_obj_n"]) == 0


def test_gtloader_part_flags_opt_in_train():
    """part_flags=True on train only adds the three flags (all False for an absent token)."""
    d = E.GTLoader(DATA, part_flags=True).load("ffffffffffffffff")
    assert list(d)[-3:] == list(E.GT_PART_KEYS) and not any(bool(d[k]) for k in E.GT_PART_KEYS)


# ----------------------------------------------------------------------------------------------- builder
def _read_npz(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.skipif(not NAVTRAIN_STATS.is_file(), reason="navtrain SDF build stats absent")
def test_same_bytes_as_navtrain(tmp_path):
    """build_one on navtrain tokens (their recorded metric caches) reproduces the stored navtrain files key for key,
    bit for bit -> the new splits share the navtrain provenance."""
    recs = []
    with open(NAVTRAIN_STATS) as fh:
        for line in fh:
            r = json.loads(line)
            if r["status"] == "built" and Path(r["mc_path"]).is_file():
                recs.append(r)
            if len(recs) == 2:
                break
    if not recs:
        pytest.skip("no navtrain metric cache")
    for r in recs:
        mc = Path(r["mc_path"])
        out = tmp_path / "navtrain" / f"{r['token']}.npz"
        rec = B.build_one((r["token"], mc.parents[2].name, str(mc.parents[3]), str(out), False))
        assert rec["status"] == "built", rec
        assert rec["n_polys"] == r["n_polys"] and rec["n_segments"] == r["n_segments"]
        a, b = _read_npz(out), _read_npz(SDF.SDF_ROOT / "navtrain" / f"{r['token']}.npz")
        assert list(a) == list(b)
        for k in a:
            assert a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes(), k


@need_val_mc
def test_build_resume_manifest(tmp_path):
    """3 real val tokens + 1 without a metric cache: built / no_mc, stats + manifest; a re-run keeps valid files
    ('exists', no new stats lines for them); a corrupt or mislabelled file is rebuilt."""
    s = _split_sample("navtrain_val", 3, seed=2)
    s = pd.concat([s, pd.DataFrame({"token": ["0123456789abcdef"], "log": [s.log[0]]})], ignore_index=True)
    tf = tmp_path / "toks.parquet"
    s.to_parquet(tf, index=False)
    root = tmp_path / "sdf"
    args = ["build", "--split", "navtrain_val", "--tokens", str(tf), "--out-root", str(root), "--workers", "2"]
    r1 = B.main(args)
    assert r1["summary"]["status"] == {"built": 3, "no_mc": 1}
    man = json.loads((root / "navtrain_val/_build/manifest.json").read_text())
    assert man["n_tokens"] == 4 and man["n_valid"] == 3 and not man["complete"]
    assert [f["token"] for f in man["failures"]] == ["0123456789abcdef"]
    assert man["failures"][0]["last_status"] == "no_mc" and man["tokens"]["override"]
    assert set(man["code_sha256"]) == {str(p) for p in B.CODE}
    for t in s.token[:3]:
        p = root / "navtrain_val" / f"{t}.npz"
        z = _read_npz(p)
        assert list(z) == ["sdf", "version", "token", "grid", "ego_xyh", "layers", "n_polys", "n_invalid"]
        assert z["sdf"].dtype == np.float16 and z["sdf"].shape == (SDF.E_H, SDF.E_W) and str(z["token"]) == t
        assert B.check_file(p, t) == "valid"
    stats = root / "navtrain_val/_build/stats.jsonl"
    n_lines = len(stats.read_text().splitlines())
    assert n_lines == 4
    assert (root / "navtrain_val/_build/build.pid").read_text().strip() == str(os.getpid())
    # resume: valid files are kept
    mt = {t: (root / "navtrain_val" / f"{t}.npz").stat().st_mtime_ns for t in s.token[:3]}
    r2 = B.main(args)
    assert r2["summary"]["status"] == {"exists": 3, "no_mc": 1}
    assert all((root / "navtrain_val" / f"{t}.npz").stat().st_mtime_ns == mt[t] for t in mt)
    assert len(stats.read_text().splitlines()) == n_lines + 1
    # a corrupt file and a file carrying another token's field are rebuilt
    t0, t1 = s.token[0], s.token[1]
    (root / "navtrain_val" / f"{t0}.npz").write_bytes(b"garbage")
    os.replace(root / "navtrain_val" / f"{t1}.npz", root / "navtrain_val" / f"{t1}.tmp")
    import shutil
    shutil.copy(root / "navtrain_val" / f"{s.token[2]}.npz", root / "navtrain_val" / f"{t1}.npz")
    assert B.check_file(root / "navtrain_val" / f"{t0}.npz", t0).startswith("invalid")
    assert B.check_file(root / "navtrain_val" / f"{t1}.npz", t1).startswith("invalid: token key")
    r3 = B.main(args)
    assert r3["summary"]["status"] == {"rebuilt": 2, "exists": 1, "no_mc": 1}
    ref = SDF.load_sdf(root / "navtrain_val" / f"{t1}.tmp")
    assert SDF.load_sdf(root / "navtrain_val" / f"{t1}.npz").tobytes() == ref.tobytes()
    assert r3["manifest"]["n_valid"] == 3


def test_split_tokens_checks():
    """The split files match SPLITS n and (val) the sorted-token sha16; the metric-cache roots are the labels' ones."""
    if not HAVE_VAL_MC:
        pytest.skip("split files absent")
    df, info = B.split_tokens("navtrain_val")
    assert len(df) == 18179 and info["sha16_sorted_tokens"] == C.SPLITS["navtrain_val"]["sha16_sorted_tokens"]
    if Path(C.SPLITS["navtest"]["tokens"]).is_file():
        assert len(B.split_tokens("navtest")[0]) == 12146
    from tools.ck.data import common as CM
    r = df.iloc[0]
    assert B.mc_path(Path(C.SPLITS["navtrain_val"]["metric_cache"]), r.log, r.token) == \
        CM.mc_path("navtrain_val", r.log, r.token)


def test_claim_pid(tmp_path):
    """A live build_sdf_split process named in the pid file blocks a second start; a dead / foreign pid does not."""
    pf = tmp_path / "build.pid"
    pf.write_text("999999999\n")
    B.claim_pid(pf)
    assert pf.read_text().strip() == str(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "build_sdf_split"])
    try:
        for _ in range(50):
            if Path(f"/proc/{p.pid}/cmdline").read_bytes().count(b"build_sdf_split"):
                break
            time.sleep(0.1)
        pf.write_text(f"{p.pid}\n")
        with pytest.raises(SystemExit):
            B.claim_pid(pf)
    finally:
        p.kill()
        p.wait()
    B.claim_pid(pf)


# ----------------------------------------------------------------------------------------------- validation helpers
def test_corner_eval_synthetic():
    """Corridor |y| < 3 m: a straight run at y = 0 passes, a drift to y = 2.2 m fails (left corners at 3.35 m) from the
    first state whose corner crosses; ego_corners_np == sdf.ego_corners; out-of-grid corners are excluded."""
    xs, ys = SDF.grid_centres()
    field = np.broadcast_to(np.clip(3.0 - np.abs(ys), -10, 10)[None, :], (SDF.E_H, SDF.E_W)).astype(np.float16)
    T = 41
    st = np.zeros((3, T, 6))
    st[:, :, 0] = np.linspace(0, 40, T)
    st[1, :, 1] = np.linspace(0, 2.2, T)
    st[2, :, 0] = np.linspace(0, 80, T)                    # runs off the front of the grid
    r = B.corner_eval(field, st)
    assert r["pred_pass"].tolist() == [True, False, True]
    y_left = st[1, :, 1] + SDF.EGO_HALF_WID
    assert r["first_t"][1] == int(np.argmax(y_left >= 3.0)) and r["first_t"][0] == -1
    assert r["n_oog"][0] == 0 and r["n_oog"][2] > 0
    assert abs(r["min_sdf"][0] - (3.0 - SDF.EGO_HALF_WID)) < 2e-3
    pose = np.array([[1.0, -2.0, 0.3], [5.0, 1.0, -2.9]])
    np.testing.assert_allclose(B.ego_corners_np(pose), SDF.ego_corners(torch.as_tensor(pose)).numpy(), atol=1e-12)
    c = B._confusion(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool))
    assert (c["both_pass"], c["miss"], c["both_fail"], c["false_alarm"], c["agree"]) == (2, 1, 1, 1, 3)


def test_npz_member(tmp_path):
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    b = np.asfortranarray(np.arange(6, dtype=np.int64).reshape(2, 3))
    np.savez(tmp_path / "u.npz", a=a, b=b, s=np.array("x"))
    np.savez_compressed(tmp_path / "c.npz", a=a)
    m = B.npz_member(tmp_path / "u.npz", "a")
    assert isinstance(m, np.memmap) and np.array_equal(m, a)
    assert np.array_equal(B.npz_member(tmp_path / "u.npz", "b"), b)
    assert str(B.npz_member(tmp_path / "u.npz", "s", mmap=False)) == "x"
    c = B.npz_member(tmp_path / "c.npz", "a")
    assert not isinstance(c, np.memmap) and np.array_equal(c, a)


@pytest.mark.skipif(not _complete("navtrain_val") or not (B.F1_OUT / "val/run.npz").is_file()
                    or not B.VAL_LABELS.is_file(), reason="sdf/navtrain_val not complete or F1 val run absent")
def test_stored_val_field_vs_official_dac():
    """60 F1 val tokens (seed 3): the stored field's corner rule reproduces the official DAC label of the v2 top-1."""
    f1 = B.F1_OUT / "val/run.npz"
    toks = B.npz_member(f1, "tokens", mmap=False).astype(str)
    rows = np.asarray(B.npz_member(f1, "rows", mmap=False))
    sim = B.npz_member(f1, "sim")
    lab = np.load(B.VAL_LABELS, mmap_mode="r")
    pick = np.sort(np.random.default_rng(3).choice(len(toks), 60, replace=False))
    pred = []
    for i in pick:
        field = SDF.load_sdf(SDF.sdf_path(str(toks[i]), "navtrain_val"))
        pred.append(bool(B.corner_eval(field, np.asarray(sim[i, :1], np.float64))["pred_pass"][0]))
    off = np.asarray(lab[rows[pick], 0, B.DAC_COL]) >= 1
    assert (np.array(pred) == off).sum() >= 59


@pytest.mark.parametrize("split", ["navtrain_val", "navtest"])
def test_real_split_manifest(split):
    """The built SDF root of the split: complete manifest (every SPLITS token valid, no failure), exactly one npz per
    token, and (when run) the official-DAC validation passed (headline v2 top-1 agreement >= 99 %)."""
    m = SDF.SDF_ROOT / split / "_build" / "manifest.json"
    if not _complete(split):
        pytest.skip(f"{m} absent or not complete (build in progress)")
    man = json.loads(m.read_text())
    toks = pd.read_parquet(C.SPLITS[split]["tokens"]).token.astype(str)
    assert man["n_tokens"] == man["n_valid"] == len(toks) == C.SPLITS[split]["n"] and man["n_failures"] == 0
    assert man["tokens"]["sha16_sorted_tokens"] == B.tokens_sha16(toks) and not man["tokens"]["override"]
    assert man["sdf_version"] == SDF.SDF_VERSION
    names = {p.name[:-4] for p in (SDF.SDF_ROOT / split).glob("*.npz")}
    assert names == set(toks)
    v = SDF.SDF_ROOT / split / "_build" / "validation_dac.json"
    if v.is_file():
        res = json.loads(v.read_text())
        assert res["pass"] and res["headline_top1_random"]["agree_frac"] >= 0.99
