"""Tests for tools/refiner/train_refiner.py (CPU smoke on a synthetic packed split; < 2 min).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_train_refiner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.adapters import load_norm  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import decode  # noqa: E402

QUIET = lambda *a, **k: None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    df, src = SY.make_sources(tmp, 10)
    RD.pack_split("train", df, tmp / "packed", sources=src, workers=1, log_fn=QUIET)
    teacher = SY.make_fake_teacher(tmp, df.token)
    return tmp, df, teacher


def _args(tmp, teacher, arm, **kw):
    a = ["--arm", arm, "--fold", "0", "--seed", "0", "--gpu", "-1", "--packed-root", str(tmp / "packed"),
         "--runs", str(tmp / "runs"), "--teacher-root", str(teacher), "--tokens-per-batch", "3", "--workers", "0",
         "--n-norm", "4", "--log-every", "1", "--surrogate", "stub", "--inner-val-frac", "0.3"]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return TR.get_parser().parse_args(a)


def test_select_rows_fold_and_inner_val(env):
    tmp, df, _ = env
    P = RD.PackedSplit("train", tmp / "packed")
    tr, va = TR.select_rows(P, 0, 0.3)
    assert 0 not in set(P.index.fold.values[np.concatenate([tr, va])])
    assert not set(P.index.log.values[tr]) & set(P.index.log.values[va])      # log-level inner validation
    assert len(tr) + len(va) == 8


@pytest.mark.parametrize("arm,sur", [("none", "stub"), ("T", "real")])
def test_smoke_train_two_steps_and_resume(env, arm, sur):
    tmp, df, teacher = env
    a = _args(tmp, teacher, arm, max_steps=2, epochs=3, max_val_batches=1, surrogate=sur)
    run = TR.train(a)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["surrogate"] == {"stub": "stub", "real": "surrogate.surrogate_terms"}[sur] and cfg["arm"] == arm
    assert cfg["param_counts"]["adapter"] == (41152 if arm == "T" else 0)
    assert 0 < cfg["pi"] < 1 and cfg["pos_weight"] == pytest.approx(min((1 - cfg["pi"]) / cfg["pi"], 10))
    log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    steps = [r for r in log if r["kind"] == "step"]
    assert len(steps) == 2 and all(np.isfinite(r["loss"]) for r in steps)
    ep = [r for r in log if r["kind"] == "epoch"]
    assert len(ep) == 1 and np.isfinite(ep[0]["val"]["loss"])
    assert (run / "ckpt_last.pt").exists() and (run / "ckpt_best.pt").exists() and not (run / "DONE").exists()
    if arm == "T":
        mean, std, info = load_norm(run / "norm.npz")
        assert mean.shape == (256,) and info["n_tokens"] == 4 and info["sha_head"] == RD.TEACHER_SHA_HEAD
    # weights moved
    ck = torch.load(run / "ckpt_last.pt")
    assert ck["step"] == 2
    net, _ = TR.load_run_model(run, "last")
    assert net.lon_head[-1].weight.abs().sum() > 0
    # resume: continues from epoch 1 with the saved step counter
    a2 = _args(tmp, teacher, arm, max_steps=4, epochs=3, max_val_batches=1, surrogate=sur)
    TR.train(a2)
    ck = torch.load(run / "ckpt_last.pt")
    assert ck["step"] == 4 and ck["epoch"] == 1


def test_full_run_writes_done(env):
    tmp, df, teacher = env
    a = _args(tmp, teacher, "none", epochs=1, run="full_none")
    run = TR.train(a)
    assert (run / "DONE").exists()
    mtime = (run / "ckpt_last.pt").stat().st_mtime
    TR.train(a)                                                  # finished run: no-op
    assert (run / "ckpt_last.pt").stat().st_mtime == mtime


def test_gate_targets_auc_and_surrogate_resolution():
    L = torch.ones(1, 3, len(RD.LABEL_COLS))
    L[0, 1, RD.LBL["dac"]] = 0.0
    L[0, 2, RD.LBL["ttc"]] = 0.0
    assert TR.gate_targets(L).tolist() == [[0, 1, 0]]
    assert TR.gate_targets(L, use_ttc=True).tolist() == [[0, 1, 1]]
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=300).round(1), rng.random(300) > 0.6
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert TR.auc(s, y) == pytest.approx(brute)
    fn, label = TR.resolve_surrogate("stub")
    assert fn is TR.stub_terms and label == "stub"
    fn, label = TR.resolve_surrogate("auto")
    assert fn is TR.surrogate_terms_batch and label == "surrogate.surrogate_terms"     # surrogate.py exists


def test_stub_terms_behaviour(env):
    tmp, df, _ = env
    P = RD.PackedSplit("train", tmp / "packed")
    b = RD.collate_tokens([RD.TokenDataset(P, [8, 9], None)[i] for i in range(2)])   # 11 / 12 m/s: reach the cone
    T, K = b["tau0"].shape[:2]
    zero = {"z_lon": torch.zeros(T, K, 6), "w_lat": torch.zeros(T, K, 6)}
    dec = TR.decode_batch(zero, b)
    t = TR.stub_terms(dec, b, T, K)
    assert set(TR.TERMS) <= set(t) and all(v.shape == (T * K,) for v in t.values())
    assert float(t["mod"].abs().max()) == 0.0 and float(t["prog"].abs().max()) == 0.0
    assert float(t["dac"].max()) == pytest.approx(0.0, abs=1e-6)            # |y| <= 0.15 on a 5 m wide road
    assert float(t["col"].max()) > 1e-3                                      # static cone on the path at x = 25 m
    # moving off the road raises the DAC term; a lateral offset raises mod
    w = torch.zeros(T, K, 6)
    w[..., 2:] = 3.0
    dec2 = TR.decode_batch({"z_lon": torch.zeros(T, K, 6), "w_lat": w}, b)
    t2 = TR.stub_terms(dec2, b, T, K)
    assert float(t2["mod"].min()) > 0.0
    # the human-overlap mask: a draft pays for the cone unless the HUMAN footprint also overlaps it
    b_far = dict(b)
    b_far["human_traj"] = b["tau0"][:, 0] + torch.tensor([0.0, 20.0, 0.0])     # human 20 m to the left: no overlap
    t_far = TR.stub_terms(dec, b_far, T, K)
    b_same = dict(b)
    b_same["human_traj"] = b["tau0"][:, 0]                                      # human = identity draft
    t_same = TR.stub_terms(dec, b_same, T, K)
    c_same, c_far = t_same["col"].reshape(T, K)[:, 0], t_far["col"].reshape(T, K)[:, 0]
    assert float(c_far.min()) > 1e-2 and bool((c_same < 0.25 * c_far).all())  # only the 0 <= g < m_col approach remains


def test_scene_from_batch_matches_surrogate_reference(env):
    """scene_from_batch (packed, torch) == surrogate.scene_from_numpy + collate_scenes (per-token npz, numpy), and so
    are the surrogate terms computed from either."""
    from navsim.agents.para_ssr.refiner import surrogate as S
    from navsim.agents.para_ssr.refiner.gt_future import load_objects
    from navsim.agents.para_ssr.refiner.sdf import load_sdf

    tmp, df, _ = env
    P = RD.PackedSplit("train", tmp / "packed")
    rows = [3, 8, 9]
    b = RD.collate_tokens([RD.TokenDataset(P, rows, None)[i] for i in range(len(rows))])
    got = TR.scene_from_batch(b, torch.float64)
    scenes = []
    for r in rows:
        tk = df.token[r]
        d = P.row(r)
        scenes.append(S.scene_from_numpy(objs=load_objects(tmp / "objects" / "train" / f"{tk}.npz"),
                                         sdf=load_sdf(tmp / "sdf" / "navtrain" / f"{tk}.npz"),
                                         centerline=d["cl_xy"][:int(d["cl_n"])].astype(np.float64),
                                         human_traj=d["human_traj"], p_pdm=d["pdm_progress_eff"], v0=d["v0"], a0=d["a0"]))
    ref = S.collate_scenes(scenes)
    for f in ("boxes", "obs", "is_agent", "human_overlap", "centerline", "cl_valid", "p_pdm", "v0", "a0", "gt_ego",
              "t_avail", "R"):
        x, y = getattr(got, f), getattr(ref, f)
        assert x.shape == y.shape, f
        if x.dtype == torch.bool:
            assert torch.equal(x, y), f
        else:
            torch.testing.assert_close(x.double(), y.double(), atol=1e-5, rtol=0, msg=f)
    assert torch.equal(got.sdf.float(), ref.sdf.float())
    T, K = b["tau0"].shape[:2]
    w = torch.zeros(T, K, 6)
    w[..., 3:] = 0.7
    z = torch.full((T, K, 6), -0.4)
    dec = decode(b["tau0"].reshape(-1, 8, 3).double(), z.reshape(-1, 6).double(), w.reshape(-1, 6).double(),
                 v0=b["v0"].double().repeat_interleave(K))
    tidx = torch.arange(T).repeat_interleave(K)
    t1 = S.surrogate_terms(dec, b["tau0"].reshape(-1, 8, 3).double(), ref, index=tidx)
    t2 = TR.surrogate_terms_batch(dec, b, T, K)
    for k in TR.TERMS + ("P1", "P0", "unknown"):
        torch.testing.assert_close(t2[k].double(), t1[k].double(), atol=1e-6, rtol=1e-6, msg=k)
    assert float(t2["prog"].max()) > 0 and float(t2["mod"].min()) > 0
