"""Tests for tools/refiner/pack_follow.py (follow-mode packing; synthetic sources, CPU, < 10 s).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_pack_follow.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools/refiner"))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
import pack_follow as PF  # noqa: E402
import refiner_synth as SY  # noqa: E402

QUIET = lambda *a, **k: None


def test_follow_picks_up_late_sources_and_completes(tmp_path):
    df, src = SY.make_sources(tmp_path, 6, missing_drafts=(4,), frame_gap_tokens=(5,))
    missing = tmp_path / "drafts" / "train" / f"{df.token[4]}.npz"
    sleeps = []

    def sleep_fn(sec):                     # the "upstream job" writes the missing draft npz between passes
        sleeps.append(sec)
        shutil.copy(tmp_path / "drafts" / "train" / f"{df.token[0]}.npz", missing)

    res = PF.follow(["train"], tmp_path / "packed", workers=1, interval=7.0, max_h=1.0, sources_fn=lambda s: src,
                    tokens_fn=lambda s: df, alive_fn=lambda: {"job": True}, sleep_fn=sleep_fn, log_fn=QUIET, chunk=2)
    assert res["reason"] == "complete" and res["passes"] == 2 and sleeps == [7.0]
    st = res["status"]["train"]
    assert st["n_usable"] == 5 and st["ready"] == 5 and st["n_frame_gap"] == 1 and st["complete"]
    P = RD.PackedSplit("train", tmp_path / "packed")
    assert P.rows_with().tolist() == [0, 1, 2, 3, 4]
    with np.load(missing) as z:
        assert np.array_equal(P.row(4)["drafts"], z["drafts"])
    assert (tmp_path / "packed" / "train" / "follow_status.json").is_file()


def test_follow_stops_when_upstream_done_without_progress(tmp_path):
    df, src = SY.make_sources(tmp_path, 4, missing_drafts=(1,))
    n_sleep = []
    res = PF.follow(["train"], tmp_path / "packed", workers=1, interval=1.0, max_h=1.0, sources_fn=lambda s: src,
                    tokens_fn=lambda s: df, alive_fn=lambda: {"job": False}, sleep_fn=lambda s: n_sleep.append(s),
                    log_fn=QUIET, chunk=2)
    # pass 1 packs what exists; pass 2 sees no progress with no upstream job alive -> stop (the draft never appears)
    assert res["reason"] == "upstream finished, no progress" and res["passes"] == 2 and len(n_sleep) == 1
    assert res["status"]["train"]["ready"] == 3 and not res["status"]["train"]["complete"]


def test_upstream_alive_detects_processes():
    alive = PF.upstream_alive(("pytest", "no_such_process_pattern_xyz_123"))
    assert alive["pytest"] and not alive["no_such_process_pattern_xyz_123"]
