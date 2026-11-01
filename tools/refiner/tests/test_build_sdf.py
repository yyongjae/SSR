"""Tests for tools/refiner/build_sdf.py (CPU, < 1 min): token-file parsing, cache lookup, build, resume.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_build_sdf.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import build_sdf as B  # noqa: E402
from navsim.agents.para_ssr.refiner import sdf as S  # noqa: E402

MC_ROOT = ROOT / "data/exp/metric_cache"
TABLE = ROOT / "report/perception_reliability/pdm_attr/table.parquet"
pytestmark = pytest.mark.skipif(not (MC_ROOT.is_dir() and TABLE.exists()), reason="navtest metric cache missing")


def _two_tokens():
    import pandas as pd
    return pd.read_parquet(TABLE, columns=["token", "log"]).iloc[[5, 7000]].reset_index(drop=True)


def test_read_tokens_formats(tmp_path):
    import pandas as pd
    df = _two_tokens()
    df.to_parquet(tmp_path / "a.parquet")
    df[["token"]].to_csv(tmp_path / "b.csv", index=False)
    (tmp_path / "c.txt").write_text("\n".join(df.token) + "\nffff000011112222\n")
    a = B.read_tokens([str(tmp_path / "a.parquet")])
    assert list(a) == list(df.token) and list(a.values()) == list(df.log)
    assert list(B.read_tokens([str(tmp_path / "b.csv")]).values()) == [None, None]
    c = B.read_tokens([str(tmp_path / "c.txt"), str(tmp_path / "a.parquet")])   # first file wins, no dupes
    assert list(c) == list(df.token) + ["ffff000011112222"] and c[df.token[0]] is None


def test_locate_with_and_without_log():
    df = _two_tokens()
    with_log = B.locate(dict(zip(df.token, df.log)), [Path("/nonexistent"), MC_ROOT])
    for t, lg in zip(df.token, df.log):
        assert with_log[t] == MC_ROOT / lg / "unknown" / t / "metric_cache.pkl"
    assert B.locate({"ffff000011112222": None}, [MC_ROOT])["ffff000011112222"] is None


def test_build_resume_and_stats(tmp_path):
    df = _two_tokens()
    tok_file = tmp_path / "toks.parquet"
    df.to_parquet(tok_file)
    (tmp_path / "extra.txt").write_text("ffff000011112222\n")
    args = ["--subset", "unit", "--tokens", str(tok_file), "--tokens", str(tmp_path / "extra.txt"),
            "--mc-root", str(MC_ROOT), "--out-root", str(tmp_path / "sdf"), "--workers", "1", "--min-age", "0"]
    s1 = B.main(args)
    assert s1["status"] == {"built": 2, "no_mc": 1}
    for t in df.token:
        f = S.load_sdf(S.sdf_path(t, "unit", tmp_path / "sdf"))
        assert f.dtype == np.float16 and f.shape == (320, 256) and np.isfinite(f).all()
        assert (f > 0).any() and (f < 0).any()
        with np.load(S.sdf_path(t, "unit", tmp_path / "sdf")) as z:
            assert str(z["token"]) == t and z["n_polys"] > 0
    lines = [json.loads(x) for x in (tmp_path / "sdf/unit/_build/stats.jsonl").read_text().splitlines()]
    assert sorted(r["status"] for r in lines) == ["built", "built"]            # transient states not logged
    assert all(r["t_build"] > 0 and r["bytes"] > 0 for r in lines if r["status"] == "built")
    # resume: existing files are skipped, the missing token is retried
    mtimes = {t: os.stat(S.sdf_path(t, "unit", tmp_path / "sdf")).st_mtime_ns for t in df.token}
    s2 = B.main(args)
    assert s2["n_existing_before"] == 2 and s2["status"] == {"no_mc": 1}
    assert all(os.stat(S.sdf_path(t, "unit", tmp_path / "sdf")).st_mtime_ns == m for t, m in mtimes.items())
    # a freshly written metric cache is left alone (a caching job may still be writing it)
    s3 = B.main(args[:-2] + ["--min-age", "1e12", "--force"])
    assert s3["status"] == {"mc_fresh": 2, "no_mc": 1}


def test_follow_gives_up_after_deadline(tmp_path):
    (tmp_path / "t.txt").write_text("ffff000011112222\n")
    s = B.main(["--subset", "unit", "--tokens", str(tmp_path / "t.txt"), "--mc-root", str(MC_ROOT),
                "--out-root", str(tmp_path / "sdf"), "--workers", "1", "--follow-interval", "0.01",
                "--follow-max-h", "0"])
    assert s["status"] == {"no_mc": 1}


def test_worker_limit():
    with pytest.raises(SystemExit):
        B.main(["--subset", "unit", "--all", "--workers", "8"])
