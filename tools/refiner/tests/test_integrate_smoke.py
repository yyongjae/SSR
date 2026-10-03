"""Tests for tools/refiner/integrate_smoke.py helpers (label source merging, untrained eval run; CPU, < 20 s).

The integration smoke itself runs on real data (report/refiner_T/integration_smoke.json); these tests pin the pieces
that decide WHAT it checks: which scored rows count as a token's labels, and that the untrained run directory used for
the "no modification = original official scores" check loads back as the identity refiner.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_integrate_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/refiner"))

import integrate_smoke as IS  # noqa: E402
import train_refiner as TR  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import decode  # noqa: E402


def _rows(tokens, k=13, error_at=None):
    rec = []
    for t in tokens:
        for i in range(k):
            rec.append(dict(token=t, k=i, nc=1.0, dac=1.0, ddc=1.0, ep=0.9, ttc=1.0, comfort=1.0, pdms=0.95,
                            raw_progress=10.0, pdm_progress_eff=12.0,
                            error="boom" if (error_at is not None and (t, i) == error_at) else None))
    return pd.DataFrame(rec)


def test_labels_table_merges_shards_and_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(IS.RD, "DATA_ROOT", tmp_path)
    sc = tmp_path / "scores"
    (sc / "dev.shards").mkdir(parents=True)
    _rows(["a", "b"]).to_parquet(sc / "dev.parquet", index=False)                    # merged file
    _rows(["b", "c"]).to_parquet(sc / "dev.shards" / "part-0.parquet", index=False)  # b duplicated, c only in a shard
    _rows(["d"], error_at=("d", 3)).to_parquet(sc / "dev.shards" / "part-1.parquet", index=False)  # scoring error
    _rows(["e"], k=12).to_parquet(sc / "dev.shards" / "part-2.parquet", index=False)  # incomplete token
    df = IS.labels_table("dev")
    assert sorted(df.token.unique()) == ["a", "b", "c"]
    assert df.groupby("token").k.nunique().eq(13).all() and len(df) == 3 * 13          # b is not counted twice
    assert "shards" in df.attrs["source"]
    # shards only (bank still scoring its first pass)
    (sc / "dev.parquet").unlink()
    assert sorted(IS.labels_table("dev").token.unique()) == ["b", "c"]


def test_untrained_run_loads_back_as_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(IS, "OUT", tmp_path)
    run = IS.write_untrained_run("none", None, {})
    net, cfg = TR.load_run_model(run, "last")
    assert cfg["arm"] == "none" and cfg["mode"] == "A" and cfg["teacher_sha_head"] == IS.RD.TEACHER_SHA_HEAD
    T, K = 2, 3
    t = 0.5 * torch.arange(1, 9, dtype=torch.float32)
    tau0 = torch.stack([torch.stack([v * t, 0.1 * k * t, torch.zeros(8)], -1) for v in (3.0, 7.0) for k in range(K)])
    tau0 = tau0.reshape(T, K, 8, 3)
    with torch.no_grad():
        o = net(None, tau0, torch.tensor([3.0, 7.0]), torch.zeros(T), torch.zeros(T, 4), torch.zeros(T, dtype=torch.long))
    assert float(o["z_lon"].abs().max()) == 0.0 and float(o["w_lat"].abs().max()) == 0.0
    dec = decode(tau0.reshape(-1, 8, 3), o["z_lon"].reshape(-1, 6), o["w_lat"].reshape(-1, 6),
                 v0=torch.tensor([3.0, 7.0]).repeat_interleave(K))
    assert dec["traj"].numpy().tobytes() == tau0.reshape(-1, 8, 3).numpy().tobytes()
