"""CK Phase 1 data tests (CPU): pack round trip, label assembly / staleness / plan compatibility, CKDataset items,
S-grid index mapping, log-balanced dump shards, lead flags, check stats; optional real-data checks (skip when the
smoke / full data are absent).

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402

SMOKE = CM.CK_DATA / "smoke"
LC = {k: i for i, k in enumerate(CM.LABEL_COLS)}


def test_navsim_pinned_and_constants():
    import navsim
    assert navsim.__file__.startswith(CM.CK)
    assert CM.CK_KEYS == ("nc", "dac", "ep", "ttc", "comfort")
    assert CM.LABEL_COLS[:5] == CM.CK_KEYS          # labels[..., :5] is the CK key order
    from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS
    assert tuple(SIM_KEYS) == ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                               "time_to_collision_within_bound", "comfort")
    s = CM.split("navtrain_val")
    assert Path(s["navsim_logs"]).is_dir() and CM.resolve_path(s["scene_filter"]).is_file()
    assert CM.split("navtest")["metric_cache"] != CM.split("navtrain_train")["metric_cache"]


# ----------------------------------------------------------------------------------------------- fake dump helpers
def _fake_plan(rng, k=16, top1_ok=True):
    anchors = rng.normal(size=(256, 8, 3)).astype(np.float32)
    off = rng.normal(scale=0.1, size=(256, 8, 3)).astype(np.float16)
    final = rng.normal(size=256).astype(np.float32)
    idx = np.argsort(-final, kind="stable")[:k]
    cand = anchors[idx] + off[idx].astype(np.float32)
    traj = cand[0].copy() if top1_ok else cand[0] + 1.0
    sim = rng.uniform(size=(5, 256)).astype(np.float32)
    im = rng.dirichlet(np.ones(256)).astype(np.float32)
    rec = {f"cand_top{k}": cand.astype(np.float32), f"cand_top{k}_idx": idx.astype(np.int16), "offset": off,
           "final": final, "im": im, "sim": sim, "model_topk_idx": idx[:6].astype(np.int16),
           "trajectory": traj.astype(np.float32), "status": rng.normal(size=8).astype(np.float32),
           "gt_traj": rng.normal(size=(8, 3)).astype(np.float32)}
    return rec


def _write_fake_dump(root: Path, split: str, toks, rng, bad=()):
    d = CM.dump_dir(split, root)
    recs = {}
    for t in toks:
        rec = _fake_plan(rng, top1_ok=t not in bad)
        CM.atomic_savez(CM.plan_path(d, t), **rec)
        CM.atomic_save_npy(CM.bev_path(d, t), rng.normal(size=(256, 50, 100)).astype(np.float16))
        recs[t] = rec
    return recs


@pytest.fixture()
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    toks = CM.split_tokens("navtest").token.tolist()[:3]
    recs = _write_fake_dump(tmp_path, "navtest", toks, rng)
    from tools.ck.data import pack_v2
    meta = pack_v2.pack("navtest", tmp_path, k=16, tokens="dumped", threads=2)
    return tmp_path, toks, recs, meta


def test_pack_round_trip(fake_root):
    root, toks, recs, meta = fake_root
    p = CM.packed_dir("navtest", root)
    t = pd.read_parquet(p / "tokens.parquet")
    order = CM.split_tokens("navtest").token.tolist()
    assert t.token.tolist() == [x for x in order if x in set(toks)] and t.row.tolist() == [0, 1, 2]
    assert meta["n"] == meta["n_ok"] == 3
    cand, idx = np.load(p / "cand.npy"), np.load(p / "cand_idx.npy")
    for r, tok in enumerate(t.token):
        z = recs[tok]
        assert np.array_equal(cand[r], z["cand_top16"]) and np.array_equal(idx[r], z["cand_top16_idx"])
        assert np.array_equal(np.load(p / "v2_final.npy")[r], z["final"][z["cand_top16_idx"]])
        assert np.array_equal(np.load(p / "v2_sim.npy")[r], z["sim"][:, z["cand_top16_idx"]].T)
        assert np.array_equal(np.load(p / "v2_im.npy")[r], z["im"][z["cand_top16_idx"]])
        assert np.array_equal(np.load(p / "sub_traj.npy")[r], z["trajectory"])
        assert np.array_equal(np.load(p / "gt_traj.npy")[r], z["gt_traj"])
        assert np.array_equal(np.load(p / "status.npy")[r], z["status"])
    assert np.load(p / "ok.npy").all()
    assert np.load(p / "cand.npy", mmap_mode="r").dtype == np.float32


def test_pack_top32_rebuild_and_missing(tmp_path):
    rng = np.random.default_rng(1)
    toks = CM.split_tokens("navtest").token.tolist()[:3]
    recs = _write_fake_dump(tmp_path, "navtest", toks, rng, bad=(toks[2],))
    from tools.ck.data import pack_v2
    import tools.ck.data.pack_v2 as P
    anchors = np.load(CM.C.ANCHORS).astype(np.float32)
    rec, why = P.read_token(toks[0], CM.dump_dir("navtest", tmp_path), 32, anchors, True)
    z = recs[toks[0]]
    order = np.argsort(-z["final"], kind="stable")[:32]
    assert why == "" and rec["cand"].shape == (32, 8, 3) and np.array_equal(rec["cand_idx"], order)
    assert np.array_equal(rec["cand"][:16], z["cand_top16"])
    assert np.allclose(rec["cand"][16:], anchors[order[16:]] + z["offset"][order[16:]].astype(np.float32))
    with pytest.raises(SystemExit):
        pack_v2.pack("navtest", tmp_path, tokens="dumped", threads=1)
    meta = pack_v2.pack("navtest", tmp_path, tokens="dumped", threads=1, allow_missing=True)
    assert meta["n_ok"] == 2 and meta["not_ok"] == {"top1_ne_trajectory": 1}
    ok = np.load(CM.packed_dir("navtest", tmp_path) / "ok.npy")
    t = pd.read_parquet(CM.packed_dir("navtest", tmp_path) / "tokens.parquet")
    assert not ok[t.token.tolist().index(toks[2])]
    assert np.isnan(np.load(CM.packed_dir("navtest", tmp_path) / "cand.npy")[~ok]).all()


# ----------------------------------------------------------------------------------------------- labels
def _fake_shards(root, split, name, cand, toks, k, crc_bad=None):
    d = CM.labels_dir(split, name, root) / "shards"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, t in enumerate(toks):
        crc = CM.traj_crc(cand[r]) if t != crc_bad else 12345
        for kk in range(k):
            v = dict(nc=1.0, dac=1.0, ep=0.5 + 0.01 * kk, ttc=float(kk % 2), comfort=1.0, ddc=1.0,
                     pdms=0.1 * r + 0.01 * kk, raw_progress=10.0, pdm_progress_eff=12.0)
            rows.append(dict(token=t, row=r, k=kk, crc=crc, **v, rec_ok=True, error="", sec_load=0.1, sec_score=0.2))
    rows.append(dict(token="zzz_not_in_split", row=99, k=-1, crc=-1, error="boom"))
    CM.atomic_to_parquet(pd.DataFrame(rows), d / "c00000.parquet")


def test_label_assemble_and_stale_guard(fake_root):
    root, toks, recs, _ = fake_root
    from tools.ck.data import label_cands as LCD
    p = CM.packed_dir("navtest", root)
    t = pd.read_parquet(p / "tokens.parquet").token.tolist()
    cand = np.load(p / "cand.npy")
    _fake_shards(root, "navtest", "cand", cand[:, :4], t[:2], 4)       # third token unscored
    src = LCD.TrajSource("npy", str(p / "cand.npy"), t, 16).open()
    # crc in the fake shards is on cand[:, :4] -> differs from the K=16 source -> stale guard fires
    with pytest.raises(SystemExit):
        LCD.assemble("navtest", "cand", root, src, 4, {})
    c4 = p / "cand4.npy"
    np.save(c4, cand[:, :4])
    src4 = LCD.TrajSource("npy", str(c4), t, 4).open()
    meta = LCD.assemble("navtest", "cand", root, src4, 4, {})
    lab = np.load(CM.labels_dir("navtest", "cand", root) / "labels.npy")
    ok = np.load(CM.labels_dir("navtest", "cand", root) / "ok.npy")
    assert lab.shape == (3, 4, 9) and ok.shape == (3, 4) and lab.dtype == np.float32
    assert ok[:2].all() and not ok[2].any() and np.isnan(lab[2]).all()
    assert np.allclose(lab[1, 3, LC["pdms"]], 0.13) and np.allclose(lab[0, 2, LC["ep"]], 0.52)
    assert meta["n_scored_tokens"] == 2 and meta["n_err_tokens"] == 1


def test_label_plan_source_compat(tmp_path):
    from tools.ck.data import label_cands as LCD
    d = tmp_path / "shards"
    base = dict(token_sha16="a", chunk=64, k=16, source_id="v2cand:navtest:k16", traj_kind="dump", traj_path="/x")
    LCD.check_plan(d, base, fresh=False)
    LCD.check_plan(d, dict(base, traj_kind="npy", traj_path="/y/cand.npy"), fresh=False)   # same source id: ok
    with pytest.raises(SystemExit):
        LCD.check_plan(d, dict(base, source_id="/other.npy"), fresh=False)
    LCD.check_plan(d, dict(base, source_id="/other.npy"), fresh=True)


# ----------------------------------------------------------------------------------------------- dataset
def _fake_kd(root, split, n, k, rng):
    kd = root / "kd_targets" / "t"
    kd.mkdir(parents=True)
    pd.read_parquet(CM.packed_dir(split, root) / "tokens.parquet").to_parquet(kd / "tokens.parquet")
    np.save(kd / "kd_score_prob.npy", rng.uniform(size=(n, k, 5)).astype(np.float32))
    np.save(kd / "kd_score_prob_corr.npy", rng.uniform(size=(n, k, 5)).astype(np.float32))
    np.save(kd / "kd_c_lon.npy", -rng.uniform(size=(n, k, 6)).astype(np.float32))
    np.save(kd / "kd_e_lat.npy", rng.normal(size=(n, k, 6)).astype(np.float32))
    np.save(kd / "kd_ok.npy", np.ones((n, k), bool))
    np.save(kd / "kd_corr_traj.npy", rng.normal(size=(n, k, 8, 3)).astype(np.float32))
    lc = CM.labels_dir(split, "kd_corr", root)
    lc.mkdir(parents=True)
    np.save(lc / "labels.npy", rng.uniform(size=(n, k, 9)).astype(np.float32))
    np.save(lc / "ok.npy", np.ones((n, k), bool))
    return kd


def test_ckdataset_items_S(fake_root):
    root, toks, recs, _ = fake_root
    rng = np.random.default_rng(2)
    p = CM.packed_dir("navtest", root)
    n = 3
    ld = CM.labels_dir("navtest", "cand", root)
    ld.mkdir(parents=True)
    lab = rng.uniform(size=(n, 16, 9)).astype(np.float32)
    np.save(ld / "labels.npy", lab)
    np.save(ld / "ok.npy", np.ones((n, 16), bool))
    kd = _fake_kd(root, "navtest", n, 16, rng)
    from tools.ck.data.ck_dataset import CKDataset, collate_ck, make_loader
    ds = CKDataset("navtest", bev="S", labels="cand", kd_dir=str(kd), corr_aug=True, lead=True, root=root)
    assert len(ds) == 3
    it = ds[1]
    t = pd.read_parquet(p / "tokens.parquet").token.tolist()
    assert it["token"] == t[1] and it["row"] == 1
    exp = {"bev": ((256, 50, 100), np.float16), "cand": ((16, 8, 3), np.float32), "status": ((8,), np.float32),
           "v2_final": ((16,), np.float32), "v2_im": ((16,), np.float32), "v2_sim": ((16, 5), np.float32),
           "gt_traj": ((8, 3), np.float32), "y": ((16, 5), np.float32), "y_ok": ((16,), np.bool_),
           "y_pdms": ((16,), np.float32), "kd_score_prob": ((16, 5), np.float32), "kd_c_lon": ((16, 6), np.float32),
           "kd_e_lat": ((16, 6), np.float32), "kd_ok": ((16,), np.bool_), "corr_traj": ((16, 8, 3), np.float32),
           "y_corr": ((16, 5), np.float32), "y_corr_ok": ((16,), np.bool_), "kd_score_prob_corr": ((16, 5), np.float32)}
    for key, (shp, dt) in exp.items():
        assert it[key].shape == shp and it[key].dtype == dt, (key, it[key].shape, it[key].dtype)
    assert it["bev_ok"] and np.isfinite(it["lead_has"])
    assert np.array_equal(it["y"], lab[1, :, :5]) and np.array_equal(it["y_pdms"], lab[1, :, LC["pdms"]])
    bev = np.load(CM.bev_path(CM.dump_dir("navtest", root), t[1]))
    assert np.array_equal(it["bev"], bev)
    b = collate_ck([ds[i] for i in range(3)])
    assert b["tokens"] == t and b["bev"].dtype == torch.float16 and b["cand"].shape == (3, 16, 8, 3)
    assert b["y_ok"].dtype == torch.bool and b["rows"].tolist() == [0, 1, 2]
    dl = make_loader(ds, tokens_per_batch=2, shuffle=True, workers=0, seed=0)
    assert sum(len(x["tokens"]) for x in dl) == 2            # drop_last on train
    ds2 = CKDataset("navtest", bev="none", labels=None, root=root, rows=np.array([2]), limit=1)
    assert len(ds2) == 1 and "bev" not in ds2[0] and "y" not in ds2[0]


def test_ckdataset_teacher_bev_real_caches(fake_root):
    root, toks, _, _ = fake_root
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    from tools.ck.data.ck_dataset import CKDataset
    if not TeacherCache.for_subset("navtest").has(toks[0]):
        pytest.skip("BEVFusion navtest cache absent")
    ds = CKDataset("navtest", bev="T", labels=None, root=root)
    it = ds[0]
    assert it["bev_ok"] and it["bev"].shape == (256, 50, 100) and it["bev"].dtype == np.float16
    ref = TeacherCache.for_subset("navtest").load_bev(it["token"], s_grid=True)
    assert np.array_equal(it["bev"], ref)
    try:
        dsm = CKDataset("navtest", bev="M", labels=None, root=root)
        im = dsm[0]
        assert im["bev"].shape == (256, 50, 100)
    except FileNotFoundError:
        pytest.skip("ReSMap navtest cache absent")
    with pytest.raises(ValueError):
        CKDataset("navtrain_val", bev="M", labels=None, root=SMOKE if (CM.packed_dir("navtrain_val", SMOKE)
                                                                        / "tokens.parquet").is_file() else root)


@pytest.mark.skipif(not (CM.packed_dir("navtrain_train", SMOKE) / "tokens.parquet").is_file(), reason="no smoke pack")
def test_ckdataset_smoke_train_gt_and_map():
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    ds = CKDataset("navtrain_train", bev="M", labels=None, gt=True, lead=True, root=SMOKE, limit=4)
    b = collate_ck([ds[i] for i in range(len(ds))])
    assert b["bev"].shape == (4, 256, 50, 100) and b["bev_ok"].all()
    assert b["ref_gt_ok"].all() and b["ref_obj_kf"].shape[0] == 4
    assert torch.isfinite(b["gt_traj"]).all()


# ----------------------------------------------------------------------------------------------- dump helpers
def test_bev_to_sgrid_index_mapping():
    from tools.ck.data.dump_v2 import bev_to_sgrid
    q = torch.arange(5000, dtype=torch.float64)[None, :, None]
    c = torch.arange(256, dtype=torch.float64)[None, None, :]
    bev = q * 1000 + c                                   # [1, 5000, 256]
    x = bev_to_sgrid(bev)
    assert x.shape == (1, 256, 50, 100)
    r, col, ch = 7, 93, 11
    assert x[0, ch, r, col] == (r * 100 + col) * 1000 + ch
    # AdapterS consumed bev_embed directly; AdapterSGrid uses bev.flatten(2).transpose(1, 2) -> same token order
    assert torch.equal(x.flatten(2).transpose(1, 2), bev)


def test_shard_tokens_partition():
    from tools.ck.data.dump_v2 import shard_tokens
    df = CM.split_tokens("navtrain_val")
    parts = [shard_tokens(df, i, 24) for i in range(24)]
    allt = [t for p in parts for t in p]
    assert len(allt) == len(df) and set(allt) == set(df.token)
    logof = dict(zip(df.token, df.log))
    owner = {}
    for i, p in enumerate(parts):
        for t in p:
            assert owner.setdefault(logof[t], i) == i      # whole logs per shard
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= df.groupby("log").size().max()
    assert shard_tokens(df, 0, 1) == df.token.tolist()


def test_claim_shard(tmp_path):
    from tools.ck.data.dump_v2 import claim_shard
    assert claim_shard(tmp_path, 0, 4, "3")
    assert not claim_shard(tmp_path, 0, 4, "3")               # held by a live pid (us)
    c = tmp_path / "claims" / "shard1of4.json"
    c.parent.mkdir(exist_ok=True)
    c.write_text(json.dumps(dict(pid=2 ** 22 + 12345, host=__import__("os").uname().nodename)))
    assert claim_shard(tmp_path, 1, 4, "3")                   # dead claim is taken over
    (tmp_path / "claims" / "shard2of4.done").write_text("x")
    assert not claim_shard(tmp_path, 2, 4, "3")


# ----------------------------------------------------------------------------------------------- lead / checks
def test_lead_finalize():
    from tools.ck.data.lead_split import finalize, OUT_COLS
    toks = CM.split_tokens("navtest")
    n = len(toks)
    src = pd.DataFrame({"token": toks.token, "log": toks.log, "has_lead": [True, False, True] * (n // 3) +
                        [False] * (n % 3), "D_1": np.nan, "D_strong": np.nan, "D_hard": np.nan, "lead_v": 3.0,
                        "lead_gap": 10.0, "thw": 1.0, "ttc0": 5.0, "censored": False})
    src.loc[0, "D_1"] = 1.0
    src.loc[2, "D_1"] = np.nan          # censored lead
    out = finalize(src.sample(frac=1.0, random_state=0), "navtest")
    assert list(out.columns) == list(OUT_COLS) and out.token.tolist() == toks.token.tolist()
    assert out.lead_decel.iat[0] == 1.0 and out.lead_decel.iat[1] == 0.0 and np.isnan(out.lead_decel.iat[2])
    with pytest.raises(AssertionError):
        finalize(src.iloc[1:], "navtest")


def test_check_label_stats_toy():
    from tools.ck.data.check_data import label_stats
    lab = np.ones((2, 3, 9), np.float32)
    lab[..., LC["pdms"]] = [[0.5, 0.9, 0.1], [0.8, 0.2, 0.3]]
    lab[0, 1, LC["nc"]] = 0.0
    ok = np.ones((2, 3), bool)
    s = label_stats(lab, ok)
    assert np.isclose(s["pdms_rank0"], 0.65) and np.isclose(s["pdms_oracle"], 0.85)
    assert np.isclose(s["fail_all_k"], 1 / 6) and s["fail_by_rank"] == [0.0, 0.5, 0.0]


# ----------------------------------------------------------------------------------------------- real data (optional)
@pytest.mark.skipif(not (CM.packed_dir("navtest", SMOKE) / "cand.npy").is_file(), reason="no smoke navtest pack")
def test_label_real_navtest_matches_csv(tmp_path):
    """Score 2 smoke navtest tokens x K=3 with the official scorer; rank 0 == v2 navtest CSV per token."""
    from tools.ck.data import label_cands as LCD
    sp = CM.packed_dir("navtest", SMOKE)
    t = pd.read_parquet(sp / "tokens.parquet").iloc[:2]
    pk = CM.packed_dir("navtest", tmp_path)
    pk.mkdir(parents=True)
    t.assign(row=[0, 1]).to_parquet(pk / "tokens.parquet", index=False)
    np.save(pk / "cand.npy", np.load(sp / "cand.npy")[:2, :3])
    meta = LCD.run("navtest", "cand", root=tmp_path, workers=2, chunk=1)
    assert meta["n_scored_tokens"] == 2
    lab = np.load(CM.labels_dir("navtest", "cand", tmp_path) / "labels.npy")
    csv = pd.read_csv(CM.NAVTEST_CSV).set_index("token").loc[t.token]
    for k, col in (("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"),
                   ("ttc", "time_to_collision_within_bound"), ("comfort", "comfort")):
        assert np.array_equal(lab[:, 0, LC[k]], csv[col].values.astype(np.float32)), k
    assert np.allclose(lab[:, 0, LC["pdms"]], csv.score.values, atol=1e-4)
