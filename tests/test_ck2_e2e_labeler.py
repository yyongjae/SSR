"""CK2 e2e T8: recording (CandRecorder2 / rec2 chunks), the 96-column labeler (tools/ck/e2e2/labeler2.py) and LabelStore2
(navsim/agents/para_ssr/ck/e2e_data2.py; SPEC ck2e2e §4).  CPU only; <= 6 labeler worker processes.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_labeler.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
from tools.ck.e2e import labeler as L1  # noqa: E402
from tools.ck.e2e2 import labeler2 as L2  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck.anchor_sampler import token_rng  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_LABEL_IDX  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import EP_TARGETS, ck_targets  # noqa: E402

CKI = list(CK_LABEL_IDX)
N_FAKE = 24
VD = Path(D2.WARMUP_DEFAULT["var_dir"])
PK = Path(D2.WARMUP_DEFAULT["packed"])


def _tokens(tmp: Path, n=N_FAKE, bad=()):
    df = pd.DataFrame(dict(token=[f"tok{i:04d}" for i in range(n)],
                           log=["nolog" if i in bad else "__fake__" for i in range(n)], row=np.arange(n)))
    p = tmp / "tokens.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def _rec(n, seed=0, shift=0.0, invalid_frac=0.1):
    g = np.random.default_rng(seed)
    traj = (g.normal(size=(n, 96, 8, 3)) + shift).astype(np.float32)
    valid = g.random((n, 96)) > invalid_frac
    valid[:, ::6] = True
    return traj, valid


def _chunk(io, epoch, rank, attempt, seq, rows, seed=0, shift=0.0):
    rows = np.asarray(rows, np.int64)
    traj, valid = _rec(len(rows), seed + 1000 * epoch + 100 * rank + seq, shift)
    p = ED.rec_chunk_path(io, epoch, rank, attempt, seq)
    D2.write_rec2_chunk(p, rows, traj, valid, np.tile(np.arange(16, dtype=np.int16), (len(rows), 1)),
                        np.ones((len(rows), 16), np.float32), np.full(len(rows), 3.0, np.float32),
                        np.arange(len(rows)), epoch, rank)
    return p, traj, valid


def _lab(io, tokens, **kw):
    kw.setdefault("workers", 2)
    kw.setdefault("poll", 0.2)
    kw.setdefault("once", True)
    kw.setdefault("scorer", "fake")
    kw.setdefault("n_rows", N_FAKE)
    kw.setdefault("status_every", 0)
    ws = kw.pop("world_size", 1)
    return L2.Labeler2(io, ws, tokens=tokens, log=lambda s: None, **kw)


def _gen(io, ep, n=N_FAKE):
    g = D2.open_generation2(io, ep, n, "r")
    return {k: np.array(g[k]) for k in D2.GEN2_KEYS}


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint8)


# ----------------------------------------------------------------------------------------------- rec2 / recorder
def test_rec2_chunk_roundtrip_and_schema(tmp_path):
    traj, valid = _rec(5)
    att = ED.new_attempt_id()
    p = ED.rec_chunk_path(tmp_path, 5, 1, att, 0)
    D2.write_rec2_chunk(p, np.arange(5) * 3, torch.from_numpy(traj), valid, np.zeros((5, 16), np.int64),
                        np.ones((5, 16)), np.arange(5), 77, epoch=5, rank=1)
    z = D2.read_rec2_chunk(p)
    assert z["traj"].dtype == np.float32 and z["valid"].dtype == np.bool_ and z["cand_idx"].dtype == np.int16
    assert int(z["fmt"]) == 2 and z["epoch_i"] == 5 and z["rank_i"] == 1
    assert np.array_equal(_bits(z["traj"]), _bits(traj)) and np.array_equal(z["valid"], valid)
    assert np.array_equal(z["gstep"], np.full(5, 77)) and z["v0"].dtype == np.float32
    assert not list(p.parent.glob(".*tmp*"))
    assert [c["name"] for c in ED.list_rec_chunks(tmp_path)] == [p.name]
    with pytest.raises(ValueError):
        D2.write_rec2_chunk(tmp_path / "x.npz", [0], traj[:1, :32], valid[:1, :32], np.zeros((1, 16)),
                            np.zeros((1, 16)), [0], 0, 0, 0)
    # a CK1 chunk is refused by the CK2 reader (and vice versa)
    p1 = ED.rec_chunk_path(tmp_path, 6, 0, att, 0)
    ED.write_rec_chunk(p1, [0], np.zeros((1, 16, 8, 3), np.float32), np.zeros((1, 16)), np.zeros((1, 16, 8, 3)),
                       np.ones((1, 16), bool), 0, 6, 0)
    with pytest.raises(ValueError):
        D2.read_rec2_chunk(p1)
    with pytest.raises((KeyError, ValueError)):                    # the CK1 reader fails on a rec2 chunk
        ED.read_rec_chunk(p)


def test_cand_recorder2(tmp_path):
    rec = D2.CandRecorder2(tmp_path, rank=1, world_size=2, epoch=4, chunk_tokens=3)
    traj, valid = _rec(8, seed=3)
    rows = np.array([0, 1, -1, 3, 4, 5, -1, 7])
    n1 = rec.add(torch.from_numpy(rows[:4]), torch.from_numpy(traj[:4]), torch.from_numpy(valid[:4]),
                 torch.zeros(4, 16, dtype=torch.long), torch.ones(4, 16), torch.full((4,), 2.0), gstep=10)
    n2 = rec.add(rows[4:], traj[4:], valid[4:], np.zeros((4, 16)), np.ones((4, 16)), np.full(4, 2.0), gstep=11)
    assert (n1, n2) == (3, 3) and len(rec.chunks) == 2 and rec.n_tokens == 6
    rec.close_epoch()
    rec.close_epoch()                                                  # idempotent
    with pytest.raises(RuntimeError):
        rec.add(rows, traj, valid, np.zeros((8, 16)), np.ones((8, 16)), np.ones(8), 0)
    done = ED.read_rec_done(tmp_path, 4)
    assert list(done) == [1] and done[1][0]["n_tokens"] == 6 and done[1][0]["chunks"] == rec.chunks
    got = [D2.read_rec2_chunk(c["path"]) for c in ED.list_rec_chunks(tmp_path, 4)]
    allrows = np.concatenate([g["row"] for g in sorted(got, key=lambda g: g["row"][0])])
    assert allrows.tolist() == [0, 1, 3, 4, 5, 7]
    keep = rows >= 0
    cat_t = np.concatenate([g["traj"] for g in sorted(got, key=lambda g: g["row"][0])])
    assert np.array_equal(_bits(cat_t), _bits(traj[keep]))
    with pytest.raises(ValueError):
        D2.CandRecorder2(tmp_path, 0, 1, 5).add([0], traj[:1, :48], valid[:1, :48], np.zeros((1, 16)),
                                                np.zeros((1, 16)), [0], 0)


# ----------------------------------------------------------------------------------------------- generation2
def test_generation2_write_rules(tmp_path):
    n = 6
    gen = D2.open_generation2(tmp_path, 3, n, "w+")
    assert np.isnan(gen["traj"]).all() and (gen["row_state"] == 0).all() and not gen["valid"].any()
    meta = json.loads((ED.gen_dir(tmp_path, 3) / "meta.json").read_text())
    assert meta["format"] == "ck2_96" and meta["k"] == 96 and meta["vnames"] == list(D2.VNAMES)
    tr = np.ones((3, 96, 8, 3), np.float32) * np.array([1, 2, 3], np.float32)[:, None, None, None]
    lab = np.ones((3, 96, 9), np.float32)
    ok = np.ones((3, 96), bool)
    va = np.zeros((3, 96), bool)
    va[:, ::2] = True
    w = D2.write_generation_rows2(gen, [2, 4, 2], tr, lab, ok, va)
    assert w.tolist() == [True, True, False]
    assert gen["traj"][2, 0, 0, 0] == 1 and gen["traj"][4, 0, 0, 0] == 2 and np.array_equal(gen["valid"][2], va[0])
    w = D2.write_generation_rows2(gen, [2], tr[:1] * 9, lab[:1], ok[:1], va[:1])
    assert not w.any() and gen["traj"][2, 0, 0, 0] == 1
    g2 = D2.open_generation2(tmp_path, 3, n, "w+")
    assert g2["row_state"].tolist() == [0, 0, 1, 0, 1, 0]
    assert ED.list_generations(tmp_path) == [3]
    with pytest.raises(FileNotFoundError):
        D2.open_generation2(tmp_path, 7, n, "r")
    with pytest.raises(ValueError):
        D2.open_generation2(tmp_path, 3, n + 1, "r")
    # a CK1 generation is refused
    ED.open_generation(tmp_path, 8, n, "w+")
    with pytest.raises(ValueError):
        D2.open_generation2(tmp_path, 8, n, "r")


# ----------------------------------------------------------------------------------------------- labeler2 (fake)
def test_labeler2_rec_to_generation(tmp_path):
    io = tmp_path / "ck_e2e2"
    tok = _tokens(tmp_path)
    p0, t0, v0 = _chunk(io, 4, 0, "aaaa0000", 0, [0, 2, 4])
    p1, t1, v1 = _chunk(io, 4, 1, "bbbb0000", 0, [1, 3, 5])
    t1[1, 10, 2, 1] = np.nan                                     # an unscorable trajectory
    D2.write_rec2_chunk(p1, [1, 3, 5], t1, v1, np.zeros((3, 16)), np.ones((3, 16)), np.ones(3), 0, 4, 1)
    ED.write_rec_done(io, 4, 0, 2, "aaaa0000", [p0.name], 3)
    assert _lab(io, tok, world_size=2).run() == 0
    assert not (ED.gen_dir(io, 4) / "DONE.json").is_file()      # rank 1 has no DONE yet
    g = _gen(io, 4)
    assert g["row_state"][:6].tolist() == [1] * 6 and g["row_state"][6:].sum() == 0
    for rows, tr, va in (([0, 2, 4], t0, v0), ([1, 3, 5], t1, v1)):
        for i, r in enumerate(rows):
            assert np.array_equal(_bits(g["traj"][r]), _bits(tr[i]))           # recorded trajectories, bitwise
            assert np.array_equal(g["valid"][r], va[i])
            fin = np.isfinite(tr[i]).all((-1, -2))
            exp = np.full((96, 9), np.nan, np.float32)
            exp[fin] = L1.fake_scores(tr[i][fin])                              # ONE call on the finite ones
            assert np.array_equal(_bits(g["labels"][r]), _bits(exp))
            assert np.array_equal(g["cand_ok"][r], fin)
    assert g["cand_ok"][3].sum() == 95 and not g["cand_ok"][3, 10]
    # invalid duplicates are labelled too (default: all 96)
    assert g["cand_ok"][0][~v0[0]].all()
    ED.write_rec_done(io, 4, 1, 2, "bbbb0000", [p1.name], 3)
    assert _lab(io, tok, world_size=2).run() == 0
    d = json.loads((ED.gen_dir(io, 4) / "DONE.json").read_text())
    assert d["n_rows_done"] == 6 and d["n_err"] == 0 and d["n_chunks"] == 2
    st = json.loads((io / "lab" / "status.json").read_text())
    assert st["epochs"]["4"]["rows_done"] == 6 and st["epochs"]["4"]["done"]       # status through open_generation2


def test_labeler2_skip_invalid(tmp_path):
    io = tmp_path / "ck_e2e2"
    tok = _tokens(tmp_path)
    _, tr, va = _chunk(io, 5, 0, "cccc0000", 0, [7])
    assert (~va[0]).sum() > 0
    assert _lab(io, tok, skip_invalid=True).run() == 0
    g = _gen(io, 5)
    assert np.array_equal(_bits(g["traj"][7]), _bits(tr[0]))       # recorded geometry kept for every column
    assert np.array_equal(g["cand_ok"][7], va[0]) and np.isnan(g["labels"][7][~va[0]]).all()
    exp = L1.fake_scores(tr[0][va[0]])
    assert np.array_equal(_bits(g["labels"][7][va[0]]), _bits(exp))


def test_labeler2_interrupted_rerun_equals_clean_run(tmp_path):
    tok = _tokens(tmp_path)
    a, b = tmp_path / "a" / "ck_e2e2", tmp_path / "b" / "ck_e2e2"
    for io in (a, b):
        for s, rows in enumerate(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11])):
            _chunk(io, 5, 0, "dddd0000", s, rows)
    assert _lab(a, tok).run() == 0
    assert _lab(b, tok, limit_chunks=1).run() == 0
    for m in (ED.gen_dir(b, 5) / "chunks").glob("*.json"):
        m.unlink()                                                   # crash between row writes and the marker
    assert _lab(b, tok).run() == 0
    ga, gb = _gen(a, 5), _gen(b, 5)
    for k in D2.GEN2_KEYS:
        assert np.array_equal(_bits(ga[k]), _bits(gb[k])), k
    ms = [json.loads(m.read_text()) for m in sorted((ED.gen_dir(b, 5) / "chunks").glob("*.json"))]
    assert len(ms) == 3 and sum(m["n_ok"] for m in ms) == 8 and sum(m["n_skip"] for m in ms) == 4
    assert _lab(b, tok).run() == 0
    gb2 = _gen(b, 5)
    assert all(np.array_equal(_bits(gb[k]), _bits(gb2[k])) for k in D2.GEN2_KEYS)


def test_labeler2_first_record_wins(tmp_path):
    io = tmp_path / "ck_e2e2"
    tok = _tokens(tmp_path)
    _, t1, v1 = _chunk(io, 7, 0, "eeee0001", 0, [3, 4])
    assert _lab(io, tok).run() == 0
    _chunk(io, 7, 0, "eeee0002", 0, [3, 4, 5], seed=9, shift=5.0)     # resumed epoch, new attempt
    assert _lab(io, tok).run() == 0
    g = _gen(io, 7)
    assert np.array_equal(_bits(g["traj"][3]), _bits(t1[0])) and np.array_equal(g["valid"][4], v1[1])
    assert g["row_state"][[3, 4, 5]].tolist() == [1, 1, 1]


def test_labeler2_errors_retry_and_bad_chunk(tmp_path, monkeypatch):
    io = tmp_path / "ck_e2e2"
    tok = _tokens(tmp_path, bad=(6,))
    fail = tmp_path / "fail.json"
    fail.write_text(json.dumps(["tok0002"]))
    monkeypatch.setenv("CK_LABELER_FAKE_FAIL", str(fail))
    p, tr, va = _chunk(io, 8, 0, "abab0000", 0, [1, 2, 6, 7])
    # a CK1 chunk in the CK2 io_dir -> BAD chunk (marker bad, errors.jsonl), nothing written for its rows
    p_bad = ED.rec_chunk_path(io, 8, 0, "abab0000", 1)
    ED.write_rec_chunk(p_bad, [9], np.zeros((1, 16, 8, 3), np.float32), np.zeros((1, 16)),
                       np.zeros((1, 16, 8, 3), np.float32), np.ones((1, 16), bool), 0, 8, 0)
    ED.write_rec_done(io, 8, 0, 1, "abab0000", [p.name, p_bad.name], 5)
    assert _lab(io, tok).run() == 0
    g = _gen(io, 8)
    assert g["row_state"][[1, 2, 6, 7, 9]].tolist() == [1, 0, 0, 1, 0]
    errs = [json.loads(x) for x in (ED.gen_dir(io, 8) / "errors.jsonl").read_text().splitlines()]
    assert sorted(e["row"] for e in errs) == [-1, 2, 6] and any("bad chunk" in e["error"] for e in errs)
    mk = json.loads((ED.gen_dir(io, 8) / "chunks" / L1.marker_name(0, p_bad.name)).read_text())
    assert mk["bad"]
    d = json.loads((ED.gen_dir(io, 8) / "DONE.json").read_text())
    assert d["n_err"] == 2 and d["n_rows_done"] == 2
    fail.write_text("[]")
    assert _lab(io, tok, retry_errors=True).run() == 0
    g = _gen(io, 8)
    assert g["row_state"][[1, 2, 6, 7]].tolist() == [1, 1, 0, 1]
    assert np.array_equal(_bits(g["traj"][2]), _bits(tr[1])) and np.array_equal(g["valid"][2], va[1])
    assert np.array_equal(_bits(g["labels"][2]), _bits(L1.fake_scores(tr[1])))
    d = json.loads((ED.gen_dir(io, 8) / "DONE.json").read_text())
    assert d["n_err"] == 1 and d["n_rows_done"] == 3


def test_labeler2_defaults_and_args(tmp_path):
    tok = _tokens(tmp_path)
    lab = L2.Labeler2(tmp_path / "io", tokens=tok, n_rows=N_FAKE, scorer="fake", log=lambda s: None)
    assert (lab.workers, lab.world_size, lab.max_epoch, lab.skip_invalid) == (32, 2, 28, False)
    with pytest.raises(AssertionError):
        _lab(tmp_path / "x=y", tok)
    with pytest.raises(AssertionError):
        _lab(tmp_path / "ok", tok, workers=49)


def test_labeler2_cli_exit_file_and_status(tmp_path):
    run = tmp_path / "RUN"
    io = run / "ck_e2e2"
    (run / "labeler").mkdir(parents=True)
    tok = _tokens(tmp_path)
    _chunk(io, 5, 0, "cafe0000", 0, [0, 1, 2])
    (io / "TRAIN_DONE").write_text("")
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, f"{REPO}/tools/ck/e2e2/labeler2.py", "--io-dir", str(io), "--world-size", "1",
                        "--workers", "2", "--poll", "0.2", "--exit-when-done", "--n-rows", str(N_FAKE),
                        "--tokens", tok, "--scorer", "fake", "--pid-file", str(run / "labeler" / "labeler.pid")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "e2e_data2" in r.stdout
    assert (run / "labeler" / "labeler.exit").read_text().strip() == "0"
    st = json.loads((io / "lab" / "status.json").read_text())
    assert st["tokens_ok"] == 3 and st["backlog_chunks"] == 0 and st["epochs"]["5"]["rows_done"] == 3
    assert st["workers"] == 2
    assert _gen(io, 5)["row_state"][:3].tolist() == [1, 1, 1]


def test_labeler2_group_sigterm_exits_143(tmp_path):
    run = tmp_path / "RUN"
    io = run / "ck_e2e2"
    (run / "labeler").mkdir(parents=True)
    tok = _tokens(tmp_path)
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    pr = subprocess.Popen([sys.executable, f"{REPO}/tools/ck/e2e2/labeler2.py", "--io-dir", str(io), "--world-size",
                           "1", "--workers", "4", "--poll", "0.2", "--status-every", "0.2", "--n-rows", str(N_FAKE),
                           "--tokens", tok, "--scorer", "fake"], env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while not (io / "lab" / "status.json").is_file() and time.time() - t0 < 60:
        time.sleep(0.2)
    time.sleep(1.0)
    os.killpg(pr.pid, signal.SIGTERM)
    try:
        rc = pr.wait(timeout=60)
    except subprocess.TimeoutExpired:                         # pragma: no cover
        os.killpg(pr.pid, signal.SIGKILL)
        pr.wait()
        pytest.fail("labeler2 hung after a process-group SIGTERM")
    left = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if os.getpgid(int(d.name)) == pr.pid:
                    left.append(int(d.name))
            except (ProcessLookupError, PermissionError):
                pass
    for q in left:                                            # only processes of the group this test started
        os.kill(q, signal.SIGKILL)
    assert rc == 143 and (run / "labeler" / "labeler.exit").read_text().strip() == "143"
    assert left == []


# ----------------------------------------------------------------------------------------------- real tokens (official)
@pytest.mark.skipif(not (VD / "labels.npy").is_file(), reason="variant label files absent")
def test_labeler2_official_equals_variant_labels(tmp_path):
    """Two real navtrain_train rows: their 96 variant-file trajectories recorded as a rec2 chunk -> labeler2 official
    scorer -> labels / ok bitwise == var_separate_sampler16_accstraight labels.npy / ok.npy (same score_token, all 96
    at once); a token without metric cache -> row_state 0 + errors.jsonl."""
    tt = pd.read_parquet(PK / "tokens.parquet")
    src = [17, 40213]
    df = pd.DataFrame(dict(token=list(tt.token.iloc[src]) + ["ffffffffffffffff"],
                           log=list(tt.log.iloc[src]) + ["no_such_log"], row=[0, 1, 2]))
    df.to_parquet(tmp_path / "tokens.parquet", index=False)
    T = np.load(VD / "traj.npy", mmap_mode="r")
    Lb = np.load(VD / "labels.npy", mmap_mode="r")
    Ok = np.load(VD / "ok.npy", mmap_mode="r")
    valid = np.load(VD / "index.npz")["valid"][src].reshape(2, 96)
    tr = np.concatenate([np.asarray(T[src], np.float32), np.zeros((1, 96, 8, 3), np.float32)])
    va = np.concatenate([valid, np.ones((1, 96), bool)])
    io = tmp_path / "ck_e2e2"
    p = ED.rec_chunk_path(io, 5, 0, "a1b2c3d4", 0)
    D2.write_rec2_chunk(p, np.arange(3), tr, va, np.zeros((3, 16)), np.zeros((3, 16)), np.zeros(3), 0, 5, 0)
    lab = L2.Labeler2(io, 1, workers=2, poll=0.2, once=True, n_rows=3, tokens=str(tmp_path / "tokens.parquet"),
                      scorer="official", status_every=0, log=lambda s: None)
    assert lab.run() == 0
    g = _gen(io, 5, 3)
    assert g["row_state"].tolist() == [1, 1, 0]
    for j, r in enumerate(src):
        want = np.asarray(Lb[r])
        assert np.array_equal(np.isnan(want), np.isnan(g["labels"][j]))
        fin = ~np.isnan(want)
        assert np.array_equal(want.view(np.uint32)[fin], g["labels"][j].view(np.uint32)[fin]), r
        assert np.array_equal(g["cand_ok"][j], np.asarray(Ok[r]))
        assert np.array_equal(_bits(g["traj"][j]), _bits(tr[j])) and np.array_equal(g["valid"][j], va[j])
    errs = [json.loads(x) for x in (ED.gen_dir(io, 5) / "errors.jsonl").read_text().splitlines()]
    assert [e["row"] for e in errs] == [2]


# ----------------------------------------------------------------------------------------------- LabelStore2
def _fill_gen2(io, e, n, rows, seed):
    g = np.random.default_rng(seed)
    gen = D2.open_generation2(io, e, n, "w+")
    m = len(rows)
    tr = g.normal(size=(m, 96, 8, 3)).astype(np.float32)
    lab = g.random((m, 96, 9)).astype(np.float32)
    lab[0, 3, 0] = np.nan                                       # an unscorable candidate
    cok = np.ones((m, 96), bool)
    cok[0, 6] = False                                           # identity of rank 1 not ok
    va = g.random((m, 96)) > 0.2
    va[:, ::6] = True
    D2.write_generation_rows2(gen, rows, tr, lab, cok, va)
    return {int(r): (tr[i], lab[i], cok[i], va[i]) for i, r in enumerate(rows)}


@pytest.mark.parametrize("ep_target", EP_TARGETS)
def test_label_store2_rules(tmp_path, ep_target):
    """G_y = ck_targets(generation labels, ep_target) (EP of ck_e2e2.ep_target; 'official' = labels[..., CK_IDX])."""
    n = 8
    io = tmp_path / "io"
    g4 = _fill_gen2(io, 4, n, [0, 1, 2, 3], seed=4)
    g5 = _fill_gen2(io, 5, n, [2, 3, 4], seed=5)
    _fill_gen2(io, 6, n, list(range(8)), seed=6)               # newer than max_epoch: never used
    gen5 = D2.open_generation2(io, 5, n, "r+")                  # half-written row (data, row_state 0) ignored
    gen5["traj"][5] = 42.0
    gen5["cand_ok"][5] = True
    gen5["traj"].flush()
    toks = [f"tok{i:04d}" for i in range(n)]
    ls = D2.LabelStore2(io, n_rows=n, ep_target=ep_target)
    assert ls.ep_target == ep_target and D2.LabelStore2(io, n_rows=n).ep_target == D2.EP_TARGET_DEFAULT
    st = ls.refresh(5)
    assert st["gens"] == [4, 5] and st["n_prev"] == 3 and st["n_older"] == 2 and st["n_fallback"] == 3
    assert st["n_phase1"] == st["n_fallback"] and st["lag_counts"] == {"1": 3, "2": 2}
    rows = [0, 2, 4, 5, 6, -1, 7]
    tk = [toks[r] if r >= 0 else "" for r in rows]
    out = ls.lookup(torch.tensor(rows), tk, epoch=6, seed=0)
    assert out["G_traj"].shape == (7, 48, 8, 3) and out["G_y"].shape == (7, 48, 5) and out["G_ok"].dtype == torch.bool
    assert out["G_src"].dtype == torch.int8 and out["G_epoch"].dtype == torch.int16 and out["G_vtype"].dtype == torch.int8
    assert out["G_src"].tolist() == [2, 1, 1, 0, 0, 0, 0] and out["G_epoch"].tolist() == [4, 5, 5, -1, -1, -1, -1]
    assert out["has"].tolist() == [True, True, True, False, False, False, False]
    for b, (r, g) in enumerate(((0, g4), (2, g5), (4, g5))):
        tr, lab, cok, va = g[r]
        cols, pad = D2.type_balanced_cols((va & cok).reshape(16, 6), 32, token_rng(toks[r], 6, 0, "ck2e2e.lab"))
        c = np.concatenate([np.arange(16) * 6, cols])
        assert np.array_equal(out["G_col"][b].numpy(), c)
        np.testing.assert_array_equal(out["G_traj"][b].numpy(), tr[c])
        yy = ck_targets(lab[c], ep_target)
        if ep_target == "official":                             # equal_nan: lab[0, 3, 0] (NC) is NaN
            assert np.array_equal(yy, lab[c][:, CKI], equal_nan=True)
        else:                                                   # random labels: r, p < 5 m -> decoupled EP = 1
            assert np.array_equal(np.delete(yy, 2, -1), np.delete(lab[c][:, CKI], 2, -1), equal_nan=True)
            assert (yy[:, 2] == 1.0).all()
        fin = np.isfinite(yy).all(-1)
        np.testing.assert_array_equal(out["G_y"][b].numpy(), np.where(fin[:, None], yy, 0))
        exp_ok = np.concatenate([cok[np.arange(16) * 6], cok[cols] & va[cols] & pad]) & fin
        np.testing.assert_array_equal(out["G_ok"][b].numpy(), exp_ok)
        assert np.array_equal(out["G_vtype"][b].numpy(), c % 6) and (out["G_vtype"][b, 16:] > 0).all()
        assert va[cols].all()                                   # only valid (& ok) variant columns drawn
    assert not bool(out["G_ok"][0, 1])                          # identity of rank 1 not ok in gen 4 row 0
    for b in (3, 4, 5, 6):
        assert not out["G_ok"][b].any() and float(out["G_traj"][b].abs().sum()) == 0.0
    # tokens from the RowMap when not given (packed), lag boundaries
    assert ls.refresh(4)["n_prev"] == 4 and ls.lookup([2], ["tok0002"], 6)["G_epoch"].tolist() == [4]
    assert ls.refresh(3)["n_fallback"] == n
    st_m1 = ls.refresh(-1)
    assert st_m1["n_prev"] == 0 and st_m1["n_older"] == 0 and st_m1["n_fallback"] == n
    assert ls.refresh(6)["n_prev"] == n
    s = ls.stats(reset=True)
    assert s["lookup_rows"] == 8 and s["no_row"] == 1 and s["src_prev"] == 3
    assert ls.stats()["lookup_rows"] == 0
    with pytest.raises(ValueError):
        ls.lookup([0], ["a", "b"], 6)
    with pytest.raises(ValueError):
        ls.lookup([0], ["a"], 6, n_orig=8)


def test_label_store2_draw_per_epoch_and_late_rows(tmp_path):
    n = 4
    io = tmp_path / "io"
    ls = D2.LabelStore2(io, n_rows=n)
    assert ls.refresh(4)["n_fallback"] == n and ls.refresh(4)["gens"] == []
    _fill_gen2(io, 4, n, [0], seed=0)
    assert ls.refresh(4)["n_prev"] == 1
    a = ls.lookup([0], ["tok0000"], epoch=5)["G_col"]
    b = ls.lookup([0], ["tok0000"], epoch=6)["G_col"]
    assert torch.equal(a[0, :16], b[0, :16]) and not torch.equal(a, b)    # identities fixed, variants per epoch
    gen = D2.open_generation2(io, 4, n, "r+")
    D2.write_generation_rows2(gen, [3], np.zeros((1, 96, 8, 3)), np.zeros((1, 96, 9)), np.ones((1, 96), bool),
                              np.ones((1, 96), bool))
    assert ls.lookup([3], ["tok0003"], 5)["G_src"].tolist() == [0]               # not before refresh
    assert ls.refresh(4)["n_prev"] == 2
    assert ls.lookup([3], ["tok0003"], 5)["G_src"].tolist() == [1]
    # a CK1 generation in the io_dir is a configuration error
    ED.open_generation(io, 2, n, "w+")
    with pytest.raises(ValueError):
        ls.refresh(4)


def test_label_supply_warning_compatible(tmp_path):
    """online.label_supply_warning (reused unchanged) on LabelStore2 stats."""
    from navsim.agents.para_ssr.ck.online import label_supply_warning
    n = 6
    io = tmp_path / "io"
    _fill_gen2(io, 4, n, [0, 1], seed=1)
    ED.write_rec_done(io, 4, 0, 2, "aaaa0000", [], 3)
    ED.write_rec_done(io, 4, 1, 2, "bbbb0000", [], 3)
    ls = D2.LabelStore2(io, n_rows=n)
    st = ls.refresh(4)
    ck = SimpleNamespace(cfg=SimpleNamespace(io_dir=str(io), record_from_epoch=4, label_lag=1), record_until=28,
                         epoch=5)
    w = label_supply_warning(ck, st)
    assert w["warn"] and w["n_rec"] == 6 and w["n_labeled"] == 2 and w["src_epoch"] == 4
    _fill_gen2(io, 4, n, [2, 3, 4], seed=2)
    w = label_supply_warning(ck, ls.refresh(4))
    assert not w["warn"] and w["n_labeled"] == 5


def _child_lookup(blob, rows, toks, q):
    ls = pickle.loads(blob)
    o = ls.lookup(torch.tensor(rows), toks, 6, 0)
    q.put({k: v.numpy() for k, v in o.items()})


def test_label_store2_pickle_child(tmp_path):
    n = 6
    io = tmp_path / "io"
    _fill_gen2(io, 4, n, [1, 2], seed=3)
    ls = D2.LabelStore2(io, n_rows=n)
    ls.refresh(4)
    rows, toks = [0, 1, 2, 5], ["tok0000", "tok0001", "tok0002", "tok0005"]
    ref = ls.lookup(torch.tensor(rows), toks, 6, 0)
    blob = pickle.dumps(ls)
    assert len(blob) < 50_000
    for ctx in ("fork", "spawn"):
        c = mp.get_context(ctx)
        q = c.Queue()
        p = c.Process(target=_child_lookup, args=(blob, rows, toks, q))
        p.start()
        got = q.get(timeout=120)
        p.join(60)
        for k, v in ref.items():
            np.testing.assert_array_equal(got[k], v.numpy(), err_msg=f"{ctx} {k}")


@pytest.mark.skipif(not (PK / "tokens.parquet").is_file(), reason="packed tokens absent")
def test_label_store2_tokens_from_rowmap(tmp_path):
    io = tmp_path / "io"
    n = 10                                                       # small generation; tokens come from the packed RowMap
    gen = D2.open_generation2(io, 4, n, "w+")
    tr, va = _rec(1, seed=8)
    D2.write_generation_rows2(gen, [5], tr, np.zeros((1, 96, 9)), np.ones((1, 96), bool), va)
    ls = D2.LabelStore2(io, n_rows=n)
    ls.refresh(4)
    tok = D2.tokens_of_rows([5])[0]
    assert tok == str(pd.read_parquet(PK / "tokens.parquet").token.iloc[5])
    a = ls.lookup([5], None, 5, 0)
    b = ls.lookup([5], [tok], 5, 0)
    assert torch.equal(a["G_col"], b["G_col"]) and bool(a["has"][0])
    assert D2.tokens_of_rows([-1, 5, 10 ** 7]) == ["", tok, ""]


# ----------------------------------------------------------------------------------------------- record -> label -> use
def test_chain_record_label_lookup(tmp_path):
    """Option B chain (SPEC §2-3): epoch e on-policy step -> online_variants -> CandRecorder2 (2 ranks) -> labeler2
    -> generation e -> epoch e+1 LabelStore2.refresh(e).lookup: G identities == the recorded top-16, G variants ==
    recorded variant columns (bitwise), labels == the scorer's labels of exactly those trajectories."""
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from navsim.agents.para_ssr.ck import cands2 as C2
    n = 8
    io = tmp_path / "ck_e2e2"
    tok = _tokens(tmp_path, n=n)
    toks = [f"tok{i:04d}" for i in range(n)]
    anc = torch.from_numpy(AS.load_anchors())
    g = torch.Generator().manual_seed(1)
    e = 6
    recs = {}
    for rank, rows in ((0, [0, 2, 4, 6]), (1, [1, 3, 5, 7])):
        top = anc[torch.randint(0, 256, (4, 16), generator=g)]
        v0 = torch.rand(4, generator=g, dtype=torch.float64) * 10
        out = C2.onpolicy_cands(top, v0, [toks[r] for r in rows], e, 0)
        V = out["V"]
        rec = D2.CandRecorder2(io, rank, 2, e, chunk_tokens=3)
        rec.add(torch.tensor(rows), V["traj96"], V["valid96"], torch.zeros(4, 16, dtype=torch.long),
                torch.ones(4, 16), v0, gstep=100)
        rec.close_epoch()
        for i, r in enumerate(rows):
            recs[r] = (V["traj96"][i].numpy(), V["valid96"][i].numpy())
    assert _lab(io, tok, world_size=2, n_rows=n).run() == 0
    assert (ED.gen_dir(io, e) / "DONE.json").is_file()
    ls = D2.LabelStore2(io, n_rows=n)
    st = ls.refresh(e + 1 - 1)                                    # epoch e+1, label_lag 1
    assert st["n_prev"] == n and st["n_fallback"] == 0
    G = ls.lookup(torch.arange(n), toks, epoch=e + 1, seed=0)
    assert G["has"].all() and (G["G_src"] == D2.SRC_PREV).all() and (G["G_epoch"] == e).all()
    for r in range(n):
        tr, va = recs[r]
        c = G["G_col"][r].numpy().astype(np.int64)
        assert np.array_equal(c[:16], np.arange(16) * 6)
        assert np.array_equal(_bits(G["G_traj"][r].numpy()), _bits(tr[c]))
        assert va[c[16:]].all()
        lab = ck_targets(L1.fake_scores(tr), ls.ep_target)
        assert np.array_equal(_bits(G["G_y"][r].numpy()), _bits(lab[c]))
        assert G["G_ok"][r].all()
