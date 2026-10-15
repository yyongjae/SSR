"""CK Phase 2 data side (navsim/agents/para_ssr/ck/e2e_data.py; contract_e2e.json data.tests 1-6)."""
from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from navsim.agents.para_ssr.ck import e2e_data as ED
from navsim.agents.para_ssr.ck.constants import CK_LABEL_IDX

P1 = ED.phase1_paths()
HAVE_P1 = (P1["packed"] / "tokens.parquet").is_file() and (P1["kd_targets"] / "kd_corr_traj.npy").is_file()
need_p1 = pytest.mark.skipif(not HAVE_P1, reason="Phase 1 files not present")
CKI = list(CK_LABEL_IDX)


def _scene(tok):
    return SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=tok))


@pytest.fixture(scope="module")
def real_rows():
    tdf = pd.read_parquet(P1["packed"] / "tokens.parquet")
    rows = [0, 41234, len(tdf) - 1]
    return [(int(r), str(tdf.token.iloc[r])) for r in rows]


@pytest.fixture(scope="module")
def builder():
    return ED.CKE2ETargetBuilder(SimpleNamespace(ck_e2e={"enabled": True}, ref_data_root=ED.REF_DATA_ROOT))


# ----------------------------------------------------------------------------------------------- (1) builder
@need_p1
def test_builder_real_tokens(builder, real_rows):
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache

    assert builder.get_unique_name() == "ck_e2e_targets"
    tc, rc = TeacherCache.for_subset("navtrain"), ResmapCache.for_subset("navtrain")
    gtl = GTLoader(ED.REF_DATA_ROOT)
    A = {n: np.load(P1[g] / f, mmap_mode="r") for n, (g, f) in ED._Phase1.FILES.items()}
    for r, tok in real_rows:
        out = builder.compute_targets(_scene(tok))
        assert out["ck_row"].dtype == torch.int64 and int(out["ck_row"]) == r
        # surrogate GT = GTLoader (no teachers)
        ref = gtl.load(tok)
        for k, v in ref.items():
            assert k in out and out[k].dtype == v.dtype and out[k].shape == v.shape, k
            assert torch.equal(out[k], v), k
        # teacher BEVs
        for i, cache in ((0, tc), (1, rc)):
            b = out[f"kd_bev_{i}"]
            assert b.dtype == torch.float16 and tuple(b.shape) == ED.BEV_SHAPE
            assert bool(out[f"kd_ok_{i}"])
            np.testing.assert_array_equal(b.numpy(), np.asarray(cache.load_bev(tok, s_grid=True), np.float16))
        # Phase 1 replay
        assert bool(out["ck_p1_ok"]) == bool(A["ok"][r])
        exp = {"cand": A["cand"][r], "y": A["y"][r][:, CKI], "y_ok": A["y_ok"][r], "kdc": A["kdc"][r],
               "y_kdc": A["y_kdc"][r][:, CKI], "y_kdc_ok": A["y_kdc_ok"][r], "kd_prob": A["kd_prob"][r],
               "kd_prob_corr": A["kd_prob_corr"][r], "kd_c_lon": A["kd_c_lon"][r], "kd_e_lat": A["kd_e_lat"][r],
               "kd_ok": A["kd_ok"][r]}
        shapes = {"cand": (16, 8, 3), "y": (16, 5), "y_ok": (16,), "kdc": (16, 8, 3), "y_kdc": (16, 5),
                  "y_kdc_ok": (16,), "kd_prob": (16, 5), "kd_prob_corr": (16, 5), "kd_c_lon": (16, 6),
                  "kd_e_lat": (16, 6), "kd_ok": (16,)}
        for n, e in exp.items():
            t = out[f"ck_p1_{n}"]
            assert tuple(t.shape) == shapes[n], n
            assert t.dtype == (torch.bool if n.endswith("ok") else torch.float32), n
            np.testing.assert_array_equal(t.numpy(), np.asarray(e), err_msg=n)


# ----------------------------------------------------------------------------------------------- (2) missing token
@need_p1
def test_builder_missing_token(builder):
    out = builder.compute_targets(_scene("ffffffffffffffff"))
    assert int(out["ck_row"]) == -1
    assert not bool(out["ck_p1_ok"]) and not bool(out["ref_gt_ok"])
    for i in (0, 1):
        assert not bool(out[f"kd_ok_{i}"]) and float(out[f"kd_bev_{i}"].float().abs().sum()) == 0.0
    for n in ED.CKE2ETargetBuilder.P1_KEYS:
        t = out[f"ck_p1_{n}"]
        assert float(t.float().abs().sum()) == 0.0, n


def test_builder_det_only_and_topk_guard():
    b = ED.CKE2ETargetBuilder(ck_cfg={"teacher_map_run": ""})
    assert b.teacher_runs == (ED.TEACHER_DET_RUN,)
    with pytest.raises(ValueError):
        ED.CKE2ETargetBuilder(ck_cfg={"topk": 8})


# ----------------------------------------------------------------------------------------------- (3) record chunks
def _rec(n=5, k=16, seed=0):
    g = np.random.default_rng(seed)
    return dict(row=np.arange(n) * 7, cand=g.normal(size=(n, k, 8, 3)).astype(np.float32),
                cand_idx=g.integers(0, 256, (n, k)), kd_corr=g.normal(size=(n, k, 8, 3)).astype(np.float32),
                kd_ok=g.random((n, k)) > 0.2, gstep=123)


def test_rec_chunk_roundtrip(tmp_path):
    a = _rec()
    att = ED.new_attempt_id()
    p = ED.rec_chunk_path(tmp_path, 5, 2, att, 3)
    assert p == tmp_path / "rec" / "ep005" / "r2" / f"c_{att}_000003.npz"
    ED.write_rec_chunk(p, a["row"], torch.from_numpy(a["cand"]), a["cand_idx"], a["kd_corr"], a["kd_ok"],
                       a["gstep"], epoch=5, rank=2)
    z = ED.read_rec_chunk(p)
    assert z["row"].dtype == np.int64 and z["cand_idx"].dtype == np.int16 and z["kd_ok"].dtype == np.bool_
    np.testing.assert_array_equal(z["row"], a["row"])
    np.testing.assert_array_equal(z["cand"], a["cand"])
    np.testing.assert_array_equal(z["cand_idx"], a["cand_idx"])
    np.testing.assert_array_equal(z["kd_corr"], a["kd_corr"])
    np.testing.assert_array_equal(z["kd_ok"], a["kd_ok"])
    np.testing.assert_array_equal(z["gstep"], np.full(5, 123))
    assert z["epoch_i"] == 5 and z["rank_i"] == 2
    assert not list(p.parent.glob(".*tmp*"))           # tmp replaced
    # a leftover tmp (killed writer) and junk are ignored by the listing
    (p.parent / f".c_{att}_000004.npz.tmp999").write_bytes(b"partial")
    (p.parent / "junk.npz").write_bytes(b"x")
    ls = ED.list_rec_chunks(tmp_path)
    assert [d["name"] for d in ls] == [p.name] and ls[0]["epoch"] == 5 and ls[0]["rank"] == 2 \
        and ls[0]["attempt"] == att and ls[0]["seq"] == 3
    assert ED.parse_rec_chunk_path(p.parent / f".c_{att}_000004.npz.tmp999") is None
    # DONE
    d = ED.write_rec_done(tmp_path, 5, 2, 4, att, [p], 5)
    assert d.name == f"DONE_{att}.json"
    done = ED.read_rec_done(tmp_path, 5)
    assert list(done) == [2] and done[2][0]["chunks"] == [p.name] and done[2][0]["world_size"] == 4


def test_rec_chunk_bad_shapes(tmp_path):
    a = _rec()
    with pytest.raises(ValueError):
        ED.write_rec_chunk(tmp_path / "x.npz", a["row"], a["cand"][:, :, :7], a["cand_idx"], a["kd_corr"],
                           a["kd_ok"], 0, 0, 0)
    with pytest.raises(ValueError):
        ED.rec_chunk_path(tmp_path, 0, 0, "XYZ", 0)


# ----------------------------------------------------------------------------------------------- (4) generations / LabelStore
def _fake_phase1(root: Path, n: int, seed: int = 1):
    g = np.random.default_rng(seed)
    pk, lc, lk, kd = (root / s for s in ("packed", "lc", "lk", "kd"))
    for d in (pk, lc, lk, kd):
        d.mkdir(parents=True)
    pd.DataFrame({"token": [f"t{i:04d}" for i in range(n)], "log": ["L"] * n, "city": [""] * n,
                  "row": np.arange(n)}).to_parquet(pk / "tokens.parquet")
    np.save(pk / "cand.npy", g.normal(size=(n, 16, 8, 3)).astype(np.float32))
    ok = np.ones(n, bool)
    ok[n - 1] = False
    np.save(pk / "ok.npy", ok)
    for d in (lc, lk):
        np.save(d / "labels.npy", g.random((n, 16, 9)).astype(np.float32))
        np.save(d / "ok.npy", g.random((n, 16)) > 0.1)
    np.save(kd / "kd_corr_traj.npy", g.normal(size=(n, 16, 8, 3)).astype(np.float32))
    return {"packed": str(pk), "labels_cand": str(lc), "labels_kd_corr": str(lk), "kd_targets": str(kd)}


def _fill_gen(io, e, n, rows, seed):
    g = np.random.default_rng(seed)
    gen = ED.open_generation(io, e, n, "w+")
    m = len(rows)
    tr = g.normal(size=(m, 32, 8, 3)).astype(np.float32)
    lab = g.random((m, 32, 9)).astype(np.float32)
    lab[0, 3, 0] = np.nan                                       # an unscorable candidate
    cok = np.ones((m, 32), bool)
    cok[0, 5] = False
    ED.write_generation_rows(gen, rows, tr, lab, cok)
    return {int(r): (tr[i], lab[i], cok[i]) for i, r in enumerate(rows)}


def test_generation_write_rules(tmp_path):
    n = 6
    gen = ED.open_generation(tmp_path, 3, n, "w+")
    assert np.isnan(gen["traj"]).all() and (gen["row_state"] == 0).all() and not gen["cand_ok"].any()
    tr = np.ones((3, 32, 8, 3), np.float32)
    lab = np.ones((3, 32, 9), np.float32)
    w = ED.write_generation_rows(gen, [2, 4, 2], tr * [[[[1]]], [[[2]]], [[[3]]]], lab, np.ones((3, 32), bool))
    assert w.tolist() == [True, True, False]                    # duplicate in one call: first wins
    assert gen["traj"][2, 0, 0, 0] == 1 and gen["traj"][4, 0, 0, 0] == 2
    w = ED.write_generation_rows(gen, [2], tr[:1] * 9, lab[:1], np.ones((1, 32), bool))
    assert not w.any() and gen["traj"][2, 0, 0, 0] == 1         # done rows are never overwritten
    # restart: 'w+' again keeps the data (idempotent), reader sees the same
    g2 = ED.open_generation(tmp_path, 3, n, "w+")
    assert g2["row_state"].tolist() == [0, 0, 1, 0, 1, 0]
    g3 = ED.open_generation(tmp_path, 3, n, "r")
    assert g3["traj"][4, 0, 0, 0] == 2
    assert ED.list_generations(tmp_path) == [3]
    with pytest.raises(FileNotFoundError):
        ED.open_generation(tmp_path, 7, n, "r")
    with pytest.raises(ValueError):
        ED.open_generation(tmp_path, 3, n + 1, "r")


def test_label_store_rules(tmp_path):
    n = 8
    p1 = _fake_phase1(tmp_path / "p1", n)
    io = tmp_path / "io"
    g4 = _fill_gen(io, 4, n, [0, 1, 2, 3], seed=4)
    g5 = _fill_gen(io, 5, n, [2, 3, 4], seed=5)
    g6 = _fill_gen(io, 6, n, [0, 1, 2, 3, 4, 5, 6, 7], seed=6)   # newer than max_epoch: must never be used
    # a half-written row of ep005 (data there, row_state 0) must be ignored
    gen5 = ED.open_generation(io, 5, n, "r+")
    gen5["traj"][5] = 42.0
    gen5["cand_ok"][5] = True
    gen5["traj"].flush()
    ls = ED.LabelStore(io, p1, n_rows=n)
    st = ls.refresh(5)
    assert st["gens"] == [4, 5] and st["n_prev"] == 3 and st["n_older"] == 2 and st["n_phase1"] == 3
    assert st["lag_counts"] == {"1": 3, "2": 2}
    rows = torch.tensor([0, 2, 4, 5, 6, -1, 7])
    out = ls.lookup(rows)
    assert out["G_traj"].shape == (7, 32, 8, 3) and out["G_y"].shape == (7, 32, 5)
    assert out["G_ok"].dtype == torch.bool and out["G_src"].dtype == torch.int8 and out["G_epoch"].dtype == torch.int16
    assert out["G_src"].tolist() == [2, 1, 1, 0, 0, 0, 0]
    assert out["G_epoch"].tolist() == [4, 5, 5, -1, -1, -1, -1]
    # generation rows: values, NaN label -> ok False & y zeroed, cand_ok respected
    for b, (r, g) in enumerate(((0, g4), (2, g5), (4, g5))):
        tr, lab, cok = g[r]
        np.testing.assert_array_equal(out["G_traj"][b].numpy(), tr)
        yy = lab[:, CKI]
        fin = np.isfinite(yy).all(-1)
        np.testing.assert_array_equal(out["G_y"][b].numpy(), np.where(fin[:, None], yy, 0))
        np.testing.assert_array_equal(out["G_ok"][b].numpy(), cok & fin)
    assert not bool(out["G_ok"][0, 3]) and not bool(out["G_ok"][0, 5])
    # Phase 1 rows = cat(cand, kd_corr), labels cand / kd_corr, ok & packed row ok
    P = {k: np.load(Path(v) / f, mmap_mode="r") for k, (v, f) in
         {"cand": (p1["packed"], "cand.npy"), "kdc": (p1["kd_targets"], "kd_corr_traj.npy"),
          "y": (p1["labels_cand"], "labels.npy"), "yk": (p1["labels_kd_corr"], "labels.npy"),
          "o": (p1["labels_cand"], "ok.npy"), "ok": (p1["labels_kd_corr"], "ok.npy")}.items()}
    for b, r in ((3, 5), (4, 6)):
        np.testing.assert_array_equal(out["G_traj"][b].numpy(), np.concatenate([P["cand"][r], P["kdc"][r]]))
        np.testing.assert_array_equal(out["G_y"][b].numpy(),
                                      np.concatenate([P["y"][r][:, CKI], P["yk"][r][:, CKI]]))
        np.testing.assert_array_equal(out["G_ok"][b].numpy(), np.concatenate([P["o"][r], P["ok"][r]]))
    assert not out["G_ok"][5].any() and float(out["G_traj"][5].abs().sum()) == 0     # row -1
    assert not out["G_ok"][6].any()                                                    # packed ok False (row 7)
    # max_epoch boundary: refresh(4) uses only ep004; refresh(3) -> all Phase 1; refresh(6) sees ep006
    assert ls.refresh(4)["n_prev"] == 4 and ls.refresh(4)["n_phase1"] == 4
    assert ls.lookup(torch.tensor([2]))["G_epoch"].tolist() == [4]
    assert ls.refresh(3)["n_phase1"] == n
    st_m1 = ls.refresh(-1)                                   # on-policy at epoch 0 (smoke): no generation
    assert st_m1["n_prev"] == 0 and st_m1["n_older"] == 0 and st_m1["n_phase1"] == n
    assert ls.refresh(6)["n_prev"] == n
    s = ls.stats(reset=True)
    assert s["lookup_rows"] == 7 and s["no_row"] == 1 and s["src_phase1"] == 3 + 0 and s["src_prev"] == 3
    assert ls.stats()["lookup_rows"] == 0


def test_label_store_late_rows(tmp_path):
    """A row the labeler finishes after a refresh appears at the next refresh with the same max_epoch."""
    n = 4
    p1 = _fake_phase1(tmp_path / "p1", n)
    io = tmp_path / "io"
    ls = ED.LabelStore(io, p1, n_rows=n)
    assert ls.refresh(4)["n_phase1"] == n and ls.refresh(4)["gens"] == []     # no generation yet
    _fill_gen(io, 4, n, [0], seed=0)
    assert ls.refresh(4)["n_prev"] == 1
    gen = ED.open_generation(io, 4, n, "r+")
    ED.write_generation_rows(gen, [3], np.zeros((1, 32, 8, 3)), np.zeros((1, 32, 9)), np.ones((1, 32), bool))
    assert ls.lookup([3])["G_src"].tolist() == [0]                             # not before refresh
    assert ls.refresh(4)["n_prev"] == 2
    assert ls.lookup([3])["G_src"].tolist() == [1]


@need_p1
def test_label_store_real_phase1_fallback(tmp_path, real_rows):
    ls = ED.LabelStore(tmp_path / "io_empty", None)
    st = ls.refresh(4)
    assert st["n_phase1"] == ED.N_ROWS
    rows = [r for r, _ in real_rows]
    out = ls.lookup(torch.tensor(rows))
    A = {n: np.load(P1[g] / f, mmap_mode="r") for n, (g, f) in ED._Phase1.FILES.items()}
    for b, r in enumerate(rows):
        np.testing.assert_array_equal(out["G_traj"][b, :16].numpy(), A["cand"][r])
        np.testing.assert_array_equal(out["G_traj"][b, 16:].numpy(), A["kdc"][r])
        np.testing.assert_array_equal(out["G_y"][b, :16].numpy(), A["y"][r][:, CKI])
        np.testing.assert_array_equal(out["G_y"][b, 16:].numpy(), A["y_kdc"][r][:, CKI])
    assert out["G_src"].tolist() == [0] * len(rows)


# ----------------------------------------------------------------------------------------------- (5) pickle / workers
class _DS(torch.utils.data.Dataset):
    def __init__(self, b, toks):
        self.b, self.toks = b, toks

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        return self.b.compute_targets(_scene(self.toks[i]))


@need_p1
def test_builder_pickle_workers(real_rows):
    b = ED.CKE2ETargetBuilder(ck_cfg={})
    toks = [t for _, t in real_rows]
    ref = [b.compute_targets(_scene(t)) for t in toks]          # opens files in the main process
    assert b._gtl is not None and b.p1._mm
    s = pickle.dumps(b)
    b2 = pickle.loads(s)
    assert b2._gtl is None and not b2.p1._mm and b2.rows._pid is None
    assert len(s) < 100_000                                     # no memmaps / index in the pickle
    for ctx in ("fork", "spawn"):
        dl = torch.utils.data.DataLoader(_DS(b, toks), batch_size=None, num_workers=2,
                                         multiprocessing_context=ctx)
        got = list(dl)
        for g, r in zip(got, ref):
            assert set(g) == set(r)
            for k in r:
                assert torch.equal(g[k], r[k]), (ctx, k)


def _child_lookup(blob, rows, q):
    ls = pickle.loads(blob)
    o = ls.lookup(torch.tensor(rows))
    q.put({k: v.numpy() for k, v in o.items()})


def test_label_store_pickle_child(tmp_path):
    n = 6
    p1 = _fake_phase1(tmp_path / "p1", n)
    io = tmp_path / "io"
    _fill_gen(io, 4, n, [1, 2], seed=3)
    ls = ED.LabelStore(io, p1, n_rows=n)
    ls.refresh(4)
    rows = [0, 1, 2, 5]
    ref = ls.lookup(torch.tensor(rows))
    blob = pickle.dumps(ls)
    assert len(blob) < 50_000
    for ctx in ("fork", "spawn"):
        c = mp.get_context(ctx)
        q = c.Queue()
        p = c.Process(target=_child_lookup, args=(blob, rows, q))
        p.start()
        got = q.get(timeout=120)
        p.join(60)
        for k, v in ref.items():
            np.testing.assert_array_equal(got[k], v.numpy(), err_msg=f"{ctx} {k}")


# ----------------------------------------------------------------------------------------------- (6) prior / row map
@need_p1
def test_phase1_label_prior_matches_train_ck(monkeypatch):
    monkeypatch.delenv("CK_DATA_ROOT", raising=False)
    from tools.ck import train_ck
    a = ED.phase1_label_prior()
    b = train_ck.label_prior("navtrain_train")
    assert a.shape == (5,)
    np.testing.assert_array_equal(a, b)


@need_p1
def test_row_map(real_rows):
    for r, tok in real_rows:
        assert ED.row_of(tok) == r
        t, log = ED.token_log(r)
        assert t == tok and isinstance(log, str) and log
    assert ED.row_of("nope") == -1
    assert len(ED._rowmap()) == ED.N_ROWS
