"""Tests for navsim/agents/para_ssr/refiner/data.py (packing, resume, loading, folds; CPU, < 1 min).

Synthetic sources (refiner_synth.py) exercise every part of pack_split; one real metric cache checks the centerline
samples; the teacher-cache guard and S-grid flip are in test_adapters.py.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.gt_future import load_objects  # noqa: E402
from navsim.agents.para_ssr.refiner.sdf import load_sdf  # noqa: E402
import refiner_synth as SY  # noqa: E402

QUIET = lambda *a, **k: None


@pytest.mark.parametrize("workers", [1, 2])
def test_pack_all_parts(tmp_path, workers):
    df, src = SY.make_sources(tmp_path, 6, missing_drafts=(4,), error_tokens=(2,), frame_gap_tokens=(5,))
    s = RD.pack_split("train", df, tmp_path / "packed", sources=src, workers=workers, chunk=2, log_fn=QUIET)
    assert s["drafts"]["done"] == 5 and s["drafts"]["missing"] == 1
    assert s["labels"]["done"] == 5 and s["labels"]["token_errors"] == 1
    for p in ("human", "objects", "sdf", "centerline"):
        assert s[p]["done"] == 6, p
    P = RD.PackedSplit("train", tmp_path / "packed")
    assert list(P.index.token) == list(df.token) and P.N == 6
    assert P.rows_with().tolist() == [0, 1, 3]                  # 2: scoring error, 4: no drafts, 5: frame gap
    assert P.rows_with(("human", "drafts")).tolist() == [0, 1, 2, 3]
    # values
    for r, tk in enumerate(df.token):
        d = P.row(r)
        if r != 4:
            with np.load(tmp_path / "drafts" / "train" / f"{tk}.npz") as z:
                assert np.array_equal(d["drafts"], z["drafts"]) and np.array_equal(d["family"], z["family"])
                assert np.array_equal(d["draft_valid"], z["valid"])
        o = load_objects(tmp_path / "objects" / "train" / f"{tk}.npz")
        n = len(o["kf"])
        assert int(d["obj_n"]) == n and np.array_equal(d["obj_kf"][:n], o["kf"]) and not d["obj_kf"][n:].any()
        assert np.array_equal(d["sdf"], load_sdf(tmp_path / "sdf" / "navtrain" / f"{tk}.npz"))
    sc = pd.read_parquet(tmp_path / "scores" / "train.parquet")
    r0 = sc[sc.token == df.token[0]].sort_values("k")
    lab = P.row(0)["labels"]
    for c in RD.LABEL_COLS:
        np.testing.assert_array_equal(lab[:, RD.LBL[c]], r0[c].to_numpy(np.float64))     # exact (float64 storage)
    assert P.row(0)["pdm_progress_eff"] == r0.pdm_progress_eff.iloc[0] and lab.dtype == np.float64
    with np.load(tmp_path / "human" / "train.npz") as z:
        assert np.array_equal(P.arrays["human_traj"][:], z["traj"]) and np.array_equal(P.arrays["cmd"][:], z["cmd"])


def test_pack_resume_and_guard(tmp_path):
    df, src = SY.make_sources(tmp_path, 5, missing_drafts=(1, 3))
    RD.pack_split("train", df, tmp_path / "packed", sources=src, parts=("human", "drafts"), workers=1, log_fn=QUIET)
    P = RD.PackedSplit("train", tmp_path / "packed")
    assert P.done[:, RD.PART_ID["drafts"]].tolist() == [1, 0, 1, 0, 1]
    before = np.array(P.arrays["drafts"][0])
    # the missing draft files appear later: only those rows are (re)written
    for i in (1, 3):
        tk = df.token[i]
        np.savez(tmp_path / "drafts" / "train" / f"{tk}.npz", drafts=np.full((13, 8, 3), i, np.float32),
                 family=np.zeros(13, np.int8), params=np.zeros((13, 6), np.float32))
    np.savez(tmp_path / "drafts" / "train" / f"{df.token[0]}.npz", drafts=np.full((13, 8, 3), 99, np.float32))
    s = RD.pack_split("train", df, tmp_path / "packed", sources=src, parts=("drafts",), workers=1, log_fn=QUIET)
    assert s["drafts"]["done"] == 2
    P = RD.PackedSplit("train", tmp_path / "packed")
    assert P.done[:, RD.PART_ID["drafts"]].tolist() == [1] * 5
    assert np.array_equal(P.arrays["drafts"][0], before)                 # done rows are not rewritten
    assert (P.arrays["drafts"][3] == 3).all() and P.arrays["draft_valid"][3].all()
    s = RD.pack_split("train", df, tmp_path / "packed", sources=src, parts=("drafts",), redo=("drafts",), workers=1,
                      log_fn=QUIET)
    assert s["drafts"]["done"] == 5 and (RD.PackedSplit("train", tmp_path / "packed").arrays["drafts"][0] == 99).all()
    # a pack written with another layout (e.g. float32 labels) is refused instead of silently reused
    lp = tmp_path / "packed" / "train" / "labels.npy"
    lp.unlink()
    np.lib.format.open_memmap(lp, mode="w+", dtype=np.float32, shape=(5, 13, len(RD.LABEL_COLS))).flush()
    with pytest.raises(ValueError, match="pack layout changed"):
        RD.pack_split("train", df, tmp_path / "packed", sources=src, parts=(), log_fn=QUIET)
    with pytest.raises(ValueError, match="different token list"):
        RD.pack_split("train", df.iloc[::-1], tmp_path / "packed", sources=src, parts=(), log_fn=QUIET)


def test_loader_batches_and_folds(tmp_path):
    df, src = SY.make_sources(tmp_path, 10)
    RD.pack_split("train", df, tmp_path / "packed", sources=src, workers=1, log_fn=QUIET)
    P = RD.PackedSplit("train", tmp_path / "packed")
    rows = P.select(exclude_folds=[0])
    assert set(P.index.fold.values[rows]) == {1, 2, 3, 4} and len(rows) == 8
    assert P.select(folds=[2]).tolist() == [2, 7]
    teacher = RD.TeacherCache(SY.make_fake_teacher(tmp_path, df.token))
    dl = RD.make_loader(P, rows, teacher, tokens_per_batch=3, shuffle=True, seed=0, workers=0)
    seen = []
    for b in dl:
        T = len(b["tokens"])
        assert T == 3 and b["tau0"].shape == (3, 13, 8, 3) and b["bev"].shape == (3, 256, 50, 100)
        assert b["bev"].dtype == torch.float16
        assert torch.equal(b["bev"][0], torch.as_tensor(teacher.load_bev(b["tokens"][0])))
        A = b["obj_kf"].shape[1]
        n = [int(P.arrays["obj_n"][r]) for r in b["rows"]]
        assert A == max(n) and b["obj_valid"].sum(1).tolist() == n
        assert b["sdf"].shape == (3, 320, 256) and b["cl_xy"].shape == (3, 700, 2) and b["cl_valid"].all()
        assert b["obj_ego_kf"].shape == (3, 11, 3)
        seen += b["tokens"]
    assert len(seen) == 6 and len(set(seen)) == 6                       # drop_last: 8 rows -> 2 batches of 3
    # epoch order is reproducible from the seed
    dl.generator.manual_seed(5)
    o1 = [t for b in dl for t in b["tokens"]]
    dl.generator.manual_seed(5)
    o2 = [t for b in dl for t in b["tokens"]]
    assert o1 == o2
    dlv = RD.make_loader(P, rows, None, tokens_per_batch=3, shuffle=False, workers=0, drop_last=False)
    assert [len(b["tokens"]) for b in dlv] == [3, 3, 2] and next(iter(dlv))["bev"] is None


def test_inner_val_logs():
    logs = [f"log_{i}" for i in range(4000)]
    a = RD.inner_val_logs(logs, 0.1)
    assert a == RD.inner_val_logs(reversed(logs), 0.1)
    assert 0.08 < len(a) / 4000 < 0.12
    assert RD.inner_val_logs(logs, 0.05) <= a


def test_centerline_samples_real_metric_cache():
    from shapely.geometry import LineString, Point
    from navsim.agents.para_ssr.refiner.surrogate import centerline_from_metric_cache

    df = pd.read_parquet(RD.DATA_ROOT / "splits" / "train.parquet").iloc[:200]
    got = 0
    for tk, lg in zip(df.token, df.log):
        p = RD.locate_metric_cache(tk, lg)
        if p is None:
            continue
        mc = RD.load_metric_cache(p)
        c = RD.centerline_samples(mc)
        ref = centerline_from_metric_cache(mc)
        n = int(c["cl_n"])
        assert n == len(ref) <= RD.CL_MAX and c["cl_valid"].sum() == n and not c["cl_valid"][n:].any()
        np.testing.assert_allclose(c["cl_xy"][:n], ref, atol=1e-4, rtol=0)          # float32 storage
        # projections of the ego box centre / a point 30 m ahead: packed crop == the crop in float64 (shapely)
        line32, line64 = LineString(c["cl_xy"][:n].astype(np.float64)), LineString(ref)
        for q in ((1.461, 0.0), (31.461, 0.0)):
            assert abs(line32.project(Point(q)) - line64.project(Point(q))) < 1e-3
        got += 1
        if got >= 2:
            break
    assert got >= 1
