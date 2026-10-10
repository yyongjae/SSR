"""CK Phase 2 background labeler tests (CPU).

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python -m pytest -q tests/test_ck_e2e_labeler.py

(1) real navtrain_train tokens: rec chunk cand = packed cand, kd_corr = Phase 1 kd_corr_traj -> generation labels /
    ok bit-identical to Phase 1 labels (cand, kd_corr); a token without metric cache -> row_state 0 + errors.jsonl
(2) world_size 2 DONE rule  (3) interrupted run + rerun = one clean run, no duplicates, first record wins
(4) error rows + --retry-errors  (5) '.tmp' files ignored  (6) --exit-when-done / pid rules, CLI exit file
(7) single-instance lock  (8) status.json
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
sys.path.insert(0, REPO)
from tools.ck.e2e import labeler as L  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402

P1 = Path("/home/external-user/ssd/yongjae_refiner/ck")
PACKED = P1 / "packed/navtrain_train"
KDC = P1 / "kd_targets/phase1/kd_corr_traj.npy"
LAB_C = P1 / "labels/navtrain_train/cand"
LAB_K = P1 / "labels/navtrain_train/kd_corr"
N_FAKE = 24


def _tokens(tmp: Path, n=N_FAKE, bad=()):
    df = pd.DataFrame(dict(token=[f"tok{i:04d}" for i in range(n)],
                           log=["nolog" if i in bad else "__fake__" for i in range(n)], row=np.arange(n)))
    p = tmp / "tokens.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def _chunk(io, epoch, rank, attempt, seq, rows, seed=0, shift=0.0):
    rng = np.random.default_rng(seed + 1000 * epoch + 100 * rank + seq)
    n = len(rows)
    cand = rng.normal(size=(n, 16, 8, 3)).astype(np.float32) + np.float32(shift)
    kdc = (cand + rng.normal(scale=0.1, size=cand.shape)).astype(np.float32)
    p = ED.rec_chunk_path(io, epoch, rank, attempt, seq)
    ED.write_rec_chunk(p, np.asarray(rows, np.int64), cand, np.tile(np.arange(16, dtype=np.int16), (n, 1)), kdc,
                       np.ones((n, 16), bool), np.arange(n, dtype=np.int64), epoch, rank)
    return p, cand, kdc


def _lab(io, tokens, **kw):
    kw.setdefault("workers", 2)
    kw.setdefault("poll", 0.2)
    kw.setdefault("once", True)
    kw.setdefault("scorer", "fake")
    kw.setdefault("n_rows", N_FAKE)
    kw.setdefault("status_every", 0)
    return L.Labeler(io, kw.pop("world_size", 1), tokens=tokens, log=lambda s: None, **kw)


def _gen(io, ep, n=N_FAKE):
    g = ED.open_generation(io, ep, n, "r")
    return {k: np.array(g[k]) for k in L.GEN_KEYS}


# ----------------------------------------------------------------------------------------------- (1) real tokens
@pytest.mark.skipif(not (PACKED / "cand.npy").is_file() or not KDC.is_file(), reason="Phase 1 data absent")
def test_real_tokens_match_phase1_labels(tmp_path):
    tt = pd.read_parquet(PACKED / "tokens.parquet")
    src = [17, 40213]                                     # two real navtrain_train rows
    df = pd.DataFrame(dict(token=list(tt.token.iloc[src]) + ["ffffffffffffffff"],
                           log=list(tt.log.iloc[src]) + ["no_such_log"], row=[0, 1, 2]))
    df.to_parquet(tmp_path / "tokens.parquet", index=False)
    cand = np.load(PACKED / "cand.npy", mmap_mode="r")
    kdc = np.load(KDC, mmap_mode="r")
    c = np.concatenate([np.asarray(cand[src], np.float32), np.zeros((1, 16, 8, 3), np.float32)])
    k = np.concatenate([np.asarray(kdc[src], np.float32), np.zeros((1, 16, 8, 3), np.float32)])
    io = tmp_path / "ck_e2e"
    p = ED.rec_chunk_path(io, 5, 0, "a1b2c3d4", 0)
    ED.write_rec_chunk(p, np.arange(3), c, np.zeros((3, 16), np.int16), k, np.ones((3, 16), bool),
                       np.zeros(3, np.int64), 5, 0)
    lab = L.Labeler(io, 1, workers=2, poll=0.2, once=True, n_rows=3, tokens=str(tmp_path / "tokens.parquet"),
                    scorer="official", status_every=0, log=lambda s: None)
    assert lab.run() == 0
    g = _gen(io, 5, 3)
    assert g["row_state"].tolist() == [1, 1, 0]
    lc = np.load(LAB_C / "labels.npy", mmap_mode="r")
    lk = np.load(LAB_K / "labels.npy", mmap_mode="r")
    oc = np.load(LAB_C / "ok.npy", mmap_mode="r")
    ok_ = np.load(LAB_K / "ok.npy", mmap_mode="r")
    for j, r in enumerate(src):
        want = np.concatenate([np.asarray(lc[r]), np.asarray(lk[r])])          # [32, 9]
        got = g["labels"][j]
        assert np.array_equal(np.isnan(want), np.isnan(got))
        fin = ~np.isnan(want)
        assert np.array_equal(want.view(np.uint32)[fin], got.view(np.uint32)[fin]), (r, np.abs(want - got).max())
        assert np.array_equal(g["cand_ok"][j], np.concatenate([oc[r], ok_[r]]))
        assert np.array_equal(g["traj"][j].view(np.uint32), np.concatenate([c[j], k[j]]).view(np.uint32))
    assert np.isnan(g["labels"][2]).all() and not g["cand_ok"][2].any()
    errs = [json.loads(x) for x in (ED.gen_dir(io, 5) / "errors.jsonl").read_text().splitlines()]
    assert [e["row"] for e in errs] == [2] and "metric_cache" in errs[0]["error"] or "No such file" in errs[0]["error"]
    mk = json.loads((ED.gen_dir(io, 5) / "chunks" / L.marker_name(0, p.name)).read_text())
    assert (mk["n"], mk["n_ok"], mk["n_err"]) == (3, 2, 1)


# ----------------------------------------------------------------------------------------------- (2) DONE rule
def test_generation_done_needs_every_rank(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    p0, _, _ = _chunk(io, 4, 0, "aaaa0000", 0, [0, 2, 4])
    p1, _, _ = _chunk(io, 4, 1, "bbbb0000", 0, [1, 3, 5])
    ED.write_rec_done(io, 4, 0, 2, "aaaa0000", [p0.name], 3)
    assert _lab(io, tok, world_size=2).run() == 0
    assert (ED.gen_dir(io, 4) / "chunks" / L.marker_name(1, p1.name)).is_file()
    assert not (ED.gen_dir(io, 4) / "DONE.json").is_file()          # rank 1 has no DONE
    ED.write_rec_done(io, 4, 1, 2, "bbbb0000", [p1.name], 3)
    p2, _, _ = _chunk(io, 4, 1, "bbbb0000", 1, [7])                 # chunk not listed in DONE but pending
    assert _lab(io, tok, world_size=2, limit_chunks=0).run() == 0
    d = json.loads((ED.gen_dir(io, 4) / "DONE.json").read_text())
    assert d["n_rows_done"] == 7 and d["n_err"] == 0 and d["n_chunks"] == 3
    assert _gen(io, 4)["row_state"].sum() == 7


def test_done_waits_for_listed_chunks(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    p0, _, _ = _chunk(io, 6, 0, "cccc0000", 0, [0, 1])
    ED.write_rec_done(io, 6, 0, 1, "cccc0000", [p0.name, "c_cccc0000_000001.npz"], 3)   # second chunk missing
    assert _lab(io, tok).run() == 0
    assert not (ED.gen_dir(io, 6) / "DONE.json").is_file()
    _chunk(io, 6, 0, "cccc0000", 1, [2])
    assert _lab(io, tok).run() == 0
    assert (ED.gen_dir(io, 6) / "DONE.json").is_file()


# ----------------------------------------------------------------------------------------------- (3) restart
def test_interrupted_rerun_equals_clean_run(tmp_path):
    tok = _tokens(tmp_path)
    a, b = tmp_path / "a" / "ck_e2e", tmp_path / "b" / "ck_e2e"
    for io in (a, b):
        for s, rows in enumerate(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11])):
            _chunk(io, 5, 0, "dddd0000", s, rows)
    assert _lab(a, tok).run() == 0                                   # clean
    assert _lab(b, tok, limit_chunks=1).run() == 0                   # "killed" after one chunk
    assert len(list((ED.gen_dir(b, 5) / "chunks").glob("*.json"))) == 1
    # crash between row writes and the marker: drop the marker of the done chunk
    for m in (ED.gen_dir(b, 5) / "chunks").glob("*.json"):
        m.unlink()
    assert _lab(b, tok).run() == 0
    ga, gb = _gen(a, 5), _gen(b, 5)
    for k in L.GEN_KEYS:
        assert np.array_equal(ga[k].view(np.uint8), gb[k].view(np.uint8)), k
    ms = [json.loads(m.read_text()) for m in sorted((ED.gen_dir(b, 5) / "chunks").glob("*.json"))]
    assert len(ms) == 3 and sum(m["n_ok"] for m in ms) == 8 and sum(m["n_skip"] for m in ms) == 4
    # rerun: nothing to do, nothing changes
    assert _lab(b, tok).run() == 0
    gb2 = _gen(b, 5)
    assert all(np.array_equal(gb[k].view(np.uint8), gb2[k].view(np.uint8)) for k in L.GEN_KEYS)


def test_first_record_wins_across_attempts(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    _, c1, k1 = _chunk(io, 7, 0, "eeee0001", 0, [3, 4])
    assert _lab(io, tok).run() == 0
    _chunk(io, 7, 0, "eeee0002", 0, [3, 4, 5], seed=9, shift=5.0)      # resumed epoch, new attempt
    assert _lab(io, tok).run() == 0
    g = _gen(io, 7)
    assert np.array_equal(g["traj"][3, :16], c1[0]) and np.array_equal(g["traj"][3, 16:], k1[1 - 1])
    assert np.array_equal(g["labels"][3], L.fake_scores(np.concatenate([c1[0], k1[0]])))
    assert g["row_state"][[3, 4, 5]].tolist() == [1, 1, 1]
    assert np.array_equal(g["labels"][5], L.fake_scores(g["traj"][5]))


def test_nonfinite_trajectory_left_unscored(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    p = ED.rec_chunk_path(io, 5, 0, "ffff0000", 0)
    cand = np.ones((1, 16, 8, 3), np.float32)
    kdc = cand.copy()
    kdc[0, 3, 2, 1] = np.nan
    ED.write_rec_chunk(p, np.array([2]), cand, np.zeros((1, 16), np.int16), kdc, np.ones((1, 16), bool),
                       np.zeros(1, np.int64), 5, 0)
    assert _lab(io, tok).run() == 0
    g = _gen(io, 5)
    assert g["row_state"][2] == 1 and g["cand_ok"][2].sum() == 31 and not g["cand_ok"][2, 19]
    assert np.isnan(g["labels"][2, 19]).all() and np.isfinite(g["labels"][2, :19]).all()


# ----------------------------------------------------------------------------------------------- (4) errors
def test_error_rows_and_retry(tmp_path, monkeypatch):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path, bad=(6,))
    fail = tmp_path / "fail.json"
    fail.write_text(json.dumps(["tok0002"]))
    monkeypatch.setenv("CK_LABELER_FAKE_FAIL", str(fail))
    p, _, _ = _chunk(io, 8, 0, "abab0000", 0, [1, 2, 6, 7])
    ED.write_rec_done(io, 8, 0, 1, "abab0000", [p.name], 4)
    assert _lab(io, tok).run() == 0
    g = _gen(io, 8)
    assert g["row_state"][[1, 2, 6, 7]].tolist() == [1, 0, 0, 1]
    errs = [json.loads(x) for x in (ED.gen_dir(io, 8) / "errors.jsonl").read_text().splitlines()]
    assert sorted(e["row"] for e in errs) == [2, 6]
    d = json.loads((ED.gen_dir(io, 8) / "DONE.json").read_text())
    assert d["n_err"] == 2 and d["n_rows_done"] == 2
    fail.write_text("[]")
    assert _lab(io, tok, retry_errors=True).run() == 0
    g = _gen(io, 8)
    assert g["row_state"][[1, 2, 6, 7]].tolist() == [1, 1, 0, 1]
    assert np.array_equal(g["labels"][2], L.fake_scores(g["traj"][2]))
    d = json.loads((ED.gen_dir(io, 8) / "DONE.json").read_text())
    assert d["n_err"] == 1 and d["n_rows_done"] == 3
    # a second retry of the still-failing row appends one more error, nothing else changes
    assert _lab(io, tok, retry_errors=True).run() == 0
    errs = [json.loads(x) for x in (ED.gen_dir(io, 8) / "errors.jsonl").read_text().splitlines()]
    assert [e["row"] for e in errs].count(6) == 3 and all(Path(e["chunk"]).is_file() for e in errs)


# ----------------------------------------------------------------------------------------------- (5) tmp files
def test_tmp_and_foreign_files_ignored(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    p, _, _ = _chunk(io, 5, 0, "1234abcd", 0, [0])
    d = p.parent
    (d / f".{p.name}.tmp999").write_bytes(b"partial")
    (d / "c_1234abcd_000001.npz.tmp5").write_bytes(b"partial")
    (d / "notes.txt").write_text("x")
    recs = ED.list_rec_chunks(io)
    assert [r["name"] for r in recs] == [p.name]
    assert _lab(io, tok).run() == 0
    assert sorted(m.name for m in (ED.gen_dir(io, 5) / "chunks").iterdir()) == [L.marker_name(0, p.name)]
    assert not (ED.gen_dir(io, 5) / "errors.jsonl").exists()


def test_max_epoch_skips_later_records(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    _chunk(io, 28, 0, "aaaa1111", 0, [0])
    _chunk(io, 29, 0, "aaaa1111", 0, [1])
    assert _lab(io, tok, max_epoch=28).run() == 0
    assert L.generation_ready(io, 28) and not L.generation_ready(io, 29)


# ----------------------------------------------------------------------------------------------- (6) exit rules
def test_exit_when_done(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    _chunk(io, 5, 0, "beef0000", 0, [0, 1])
    (io / "TRAIN_DONE").write_text("")
    assert _lab(io, tok, once=False, exit_when_done=True).run() == 0      # backlog first, then exit 0
    assert _gen(io, 5)["row_state"][[0, 1]].tolist() == [1, 1]
    (io / "TRAIN_DONE").unlink()
    dead = subprocess.Popen(["true"])
    dead.wait()
    pidf = tmp_path / "train.pid"
    pidf.write_text(f"{dead.pid}\n")
    _chunk(io, 6, 0, "beef0000", 0, [2])
    lab = _lab(io, tok, once=False, exit_when_done=True, train_pid_file=str(pidf))
    assert lab.run() == 2 and lab.exit_reason == "train pid dead"
    assert _gen(io, 6)["row_state"][2] == 1
    pidf.write_text(f"{os.getpid()}\n")
    assert _lab(io, tok, train_pid_file=str(pidf)).train_finished() is None
    lab = _lab(io, tok, train_pid_file=str(tmp_path / "absent.pid"), pid_grace=0.3)
    assert lab.train_finished() is None                                   # missing, within grace
    time.sleep(0.4)
    assert lab.train_finished() == 2                                      # missing beyond grace


def test_cli_exit_file_and_status(tmp_path):
    run = tmp_path / "RUN"
    io = run / "ck_e2e"
    (run / "labeler").mkdir(parents=True)
    tok = _tokens(tmp_path)
    _chunk(io, 5, 0, "cafe0000", 0, [0, 1, 2])
    (io / "TRAIN_DONE").write_text("")
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, f"{REPO}/tools/ck/e2e/labeler.py", "--io-dir", str(io),
                        "--world-size", "1", "--workers", "2", "--poll", "0.2", "--exit-when-done",
                        "--n-rows", str(N_FAKE), "--tokens", tok, "--scorer", "fake",
                        "--pid-file", str(run / "labeler" / "labeler.pid")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (run / "labeler" / "labeler.exit").read_text().strip() == "0"
    assert (run / "labeler" / "labeler.pid").read_text().strip().isdigit()
    st = json.loads((io / "lab" / "status.json").read_text())
    assert st["tokens_ok"] == 3 and st["backlog_chunks"] == 0 and st["epochs"]["5"]["rows_done"] == 3
    assert st["recent_tok_s"] is not None and st["recent_tok_s"] > 0
    for k in ("total_tok_s", "lag_s_recent", "workers", "inflight", "backlog_tokens_est"):
        assert k in st


def _group_term(tmp_path, labeler_py=f"{REPO}/tools/ck/e2e/labeler.py", timeout=60):
    """Idle labeler (workers blocked in the pool's task queue), TERM to its whole process group (what
    train_e2e.sh stop / a group kill does): must exit 143 promptly and leave no child behind."""
    import signal
    import time
    run = tmp_path / "RUN"
    io = run / "ck_e2e"
    (run / "labeler").mkdir(parents=True)
    tok = _tokens(tmp_path)
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pr = subprocess.Popen([sys.executable, labeler_py, "--io-dir", str(io), "--world-size", "1", "--workers", "8",
                           "--poll", "0.2", "--status-every", "0.2", "--n-rows", str(N_FAKE), "--tokens", tok,
                           "--scorer", "fake"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True)
    t0 = time.time()
    while not (io / "lab" / "status.json").is_file() and time.time() - t0 < 60:
        time.sleep(0.2)
    time.sleep(1.0)
    os.killpg(pr.pid, signal.SIGTERM)
    try:
        rc = pr.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL)
        pr.wait()
        return None, run
    left = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if os.getpgid(int(d.name)) == pr.pid:
                    left.append(int(d.name))
            except (ProcessLookupError, PermissionError):
                pass
    for q in left:
        os.kill(q, signal.SIGKILL)
    return (rc, left), run


def test_group_sigterm_exits_143_without_deadlock(tmp_path):
    res, run = _group_term(tmp_path)
    assert res is not None, "labeler hung after a process-group SIGTERM"
    rc, left = res
    assert rc == 143 and (run / "labeler" / "labeler.exit").read_text().strip() == "143"
    assert left == [], f"children left behind: {left}"


# ----------------------------------------------------------------------------------------------- (7) lock
def test_single_instance_lock(tmp_path):
    io = tmp_path / "ck_e2e"
    tok = _tokens(tmp_path)
    (io / "lab").mkdir(parents=True)
    with open(io / "lab" / ".labeler.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _lab(io, tok).run() == 3
        fcntl.flock(f, fcntl.LOCK_UN)
    assert _lab(io, tok).run() == 0


def test_rejects_bad_args(tmp_path):
    tok = _tokens(tmp_path)
    with pytest.raises(AssertionError):
        _lab(tmp_path / "x=y", tok)
    with pytest.raises(AssertionError):
        _lab(tmp_path / "ok", tok, workers=49)
    with pytest.raises(AssertionError):
        _lab(tmp_path / "ok", tok, n_rows=N_FAKE + 1)                    # token table shorter than n_rows
