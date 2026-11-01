"""tools/refiner/refine_external_drafts.py: one external draft per token == the eval_refiner bank path for that draft
(CPU, synthetic split, arms none and T).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_refine_external_drafts.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
import eval_refiner as EV  # noqa: E402
import refine_external_drafts as RX  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402


@pytest.mark.parametrize("arm", ["none", "T"])
def test_single_draft_matches_bank(tmp_path, arm):
    df, src = SY.make_sources(tmp_path, 20)
    packed_root = tmp_path / "packed"
    RD.pack_split("train", df, packed_root, sources=src, workers=1, log_fn=lambda *a, **k: None)
    extra = []
    if arm == "T":
        teacher = SY.make_fake_teacher(tmp_path, df.token)
        extra = ["--teacher-root", str(teacher), "--n-norm", "4"]
    a = TR.get_parser().parse_args(["--arm", arm, "--fold", "0", "--gpu", "-1", "--packed-root", str(packed_root),
                                    "--runs", str(tmp_path / "runs"), "--tokens-per-batch", "3", "--workers", "0",
                                    "--surrogate", "stub", "--max-steps", "3", "--inner-val-frac", "0.3",
                                    "--log-every", "100"] + extra)
    run = TR.train(a)
    ev = EV.get_parser().parse_args(["predict", "--run", str(run), "--split", "train", "--fold", "0", "--packed-root",
                                     str(packed_root), "--loader-workers", "0", "--out", str(tmp_path / "bank")]
                                    + (["--teacher-root", str(teacher)] if arm == "T" else []))
    B = np.load(EV.predict(ev) / "pred.npz")
    toks = [str(t) for t in B["tokens"]]
    assert len(toks) >= 4
    for k in (0, 5):  # identity (human) slot and a perturbed slot
        pkl = tmp_path / f"drafts_k{k}.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"trajectories": {t: B["tau0"][i, k] for i, t in enumerate(toks)}, "meta": {}}, f)
        args = ["predict", "--run", str(run), "--drafts-pkl", str(pkl), "--out", str(tmp_path / f"ext{k}"), "--split",
                "train", "--packed-root", str(packed_root), "--loader-workers", "0", "--tokens-per-batch", "2"]
        if arm == "T":
            args += ["--teacher-root", str(teacher)]
        out = RX.predict(RX.get_parser().parse_args(args))
        X = np.load(out / "pred.npz")
        assert [str(t) for t in X["tokens"]] == toks and np.array_equal(X["rows"], B["rows"])
        assert X["tau1"].shape == (len(toks), 1, 8, 3)
        assert np.array_equal(X["tau0"][:, 0], B["tau0"][:, k])
        for key in ("tau1", "p_g", "z_lon", "w_lat"):
            np.testing.assert_allclose(X[key][:, 0], B[key][:, k], rtol=0, atol=1e-5, err_msg=key)
        R = np.load(out / "refined.npz")
        assert np.array_equal(R["drafts"], X["tau1"].astype(np.float32))
        tp = pd.read_parquet(out / "tokens.parquet")
        assert list(tp.token) == toks and "frame_gap" in tp.columns
    # 'original' writes the untouched drafts for the same rows
    o = RX.original(RX.get_parser().parse_args(["original", "--drafts-pkl", str(tmp_path / "drafts_k0.pkl"), "--out",
                                                str(tmp_path / "orig"), "--split", "train", "--packed-root",
                                                str(packed_root)]))
    R = np.load(o / "refined.npz")
    assert [str(t) for t in R["tokens"]] == toks and np.array_equal(R["drafts"][:, 0], B["tau0"][:, 0])


def test_load_drafts_pkl_formats(tmp_path):
    d = {"a": np.zeros((8, 3), np.float64), "b": np.ones((8, 3), np.float32)}
    for obj in (d, {"trajectories": d, "meta": {"x": 1}}):
        p = tmp_path / "d.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        got = RX.load_drafts_pkl(p)
        assert set(got) == {"a", "b"} and got["a"].dtype == np.float32
    with open(p, "wb") as f:
        pickle.dump({"a": np.zeros((7, 3))}, f)
    with pytest.raises(ValueError):
        RX.load_drafts_pkl(p)
