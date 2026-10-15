"""CK2 e2e T3: data side (navsim/agents/para_ssr/ck/e2e_data2.py, cands2.warmup_cands / fill_fallback; SPEC ck2e2e §1-4,
§2-1, §4-3 fallback, NU09 / NU12 / NU20 / NU40 / NU44).  CPU only; real files under /home/external-user/ssd/yongjae_refiner/ck.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_data.py

Also builds (once) the 2-token CK2 real-batch fixture for the model / agent tests (NU40):
  /home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt
  = features + v2 targets of the read-only CK1 fixture (packed rows 0, 40000) with every CK1 target key (ck_row, ref_*,
    kd_*, ck_p1_*) replaced by CK2E2ETargetBuilder output (ck_row, ref_*, kd_bev_0/1 + kd_ok_0/1 with the ck2T / ck2M
    teacher runs, ck2_*).  make_ck2_fixture() is importable: `from test_ck2_e2e_data import make_ck2_fixture, FIX2`.
    (The fixture file of 2026-10-07 holds official-EP ck2_raw_y / ck2_var_y; the model tests only use it as input.)
EP target (user 2026-10-08): the builder / prior / label store turn the 9 label columns into CK targets with
ep_target.ck_targets(labels, ck_e2e2.ep_target) (default 'official' since 2026-10-09; was 'decoupled'); the builder fixtures read the teacher BEVs through
the yaml-default teacher runs when they exist, else the finished official ck2T / ck2M (same arms T / M = same caches).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import cands2 as C2  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck.anchor_sampler import token_rng  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_LABEL_IDX  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import ck_targets  # noqa: E402

CKI = list(CK_LABEL_IDX)
_CKD = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train")


def _runs(det=D2.TEACHER_DET_RUN, mp=D2.TEACHER_MAP_RUN):
    """teacher runs for the BEV caches (GTLoader reads only their arm): the defaults when their config.json exists (the
    ck2T10dep / ck2M10dep runs once started), else the finished ck2T / ck2M (arms T / M)."""
    if (Path(det) / "config.json").is_file() and (Path(mp) / "config.json").is_file():
        return str(det), str(mp)
    return str(_CKD / "ck2T"), str(_CKD / "ck2M")


TDET, TMAP = _runs()
W = D2.WARMUP_DEFAULT
PK, RL, VD = Path(W["packed"]), Path(W["raw_labels"]), Path(W["var_dir"])
HAVE = all((p / f).is_file() for p, f in ((PK, "tokens.parquet"), (RL, "labels.npy"), (VD, "traj.npy"),
                                         (VD, "index.npz")))
need = pytest.mark.skipif(not HAVE, reason="packed / raw256 / variant label files absent")
FIX1 = Path("/workspace/yongjae/ssd/yongjae_refiner/ck/phase2/impl-model/fixtures/real_b2.pt")     # read-only CK1
FIX2 = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt")
CK1_TARGET_PREFIX = ("ck_row", "ref_", "kd_bev_", "kd_ok_", "ck_p1_")
torch.set_num_threads(1)


def _scene(tok):
    return SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=tok))


@pytest.fixture(scope="module")
def toks():
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rows = [0, 40000, len(tdf) - 1]
    return [(int(r), str(tdf.token.iloc[r])) for r in rows]


@pytest.fixture(scope="module")
def builder():
    return D2.CK2E2ETargetBuilder({"enabled": True, "teacher_det_run": TDET, "teacher_map_run": TMAP})


@pytest.fixture(scope="module")
def ds():
    """the teacher dataset with the student's default EP target (labels of the two sides must agree)"""
    from tools.ck.data import ck2_dataset as CD
    return CD.CK2Dataset("navtrain_train", bev="none", n_var=32, var_name=VD.name, seed=0, sampler_seed=0,
                         ep_target=D2.EP_TARGET_DEFAULT)


SHAPES = {"ck2_wu_ok": ((), torch.bool), "ck2_gt": ((8, 3), torch.float32), "ck2_raw_y": ((256, 5), torch.float32),
          "ck2_raw_ok": ((256,), torch.bool), "ck2_var_traj": ((96, 8, 3), torch.float32),
          "ck2_var_y": ((96, 5), torch.float32), "ck2_var_ok": ((96,), torch.bool),
          "ck2_var_valid": ((96,), torch.bool), "ck2_var_anchor": ((16,), torch.int64)}


# ----------------------------------------------------------------------------------------------- builder
@need
def test_builder_real_rows(builder, toks):
    from navsim.agents.para_ssr.refiner.e2e import GTLoader

    assert builder.get_unique_name() == "ck_e2e2_targets"
    chk = builder.check()
    assert chk["n_rows"] == D2.N_ROWS
    gtl = GTLoader(D2.REF_DATA_ROOT, (TDET, TMAP))
    assert builder.ep_target == D2.EP_TARGET_DEFAULT == "official"
    b_dec = D2.CK2E2ETargetBuilder({"enabled": True, "teacher_det_run": TDET, "teacher_map_run": TMAP,
                                    "ep_target": "decoupled"})
    assert b_dec.ep_target == "decoupled"
    A = {"gt": np.load(PK / "gt_traj.npy", mmap_mode="r"), "ry": np.load(RL / "labels.npy", mmap_mode="r"),
         "rok": np.load(RL / "ok.npy", mmap_mode="r"), "vt": np.load(VD / "traj.npy", mmap_mode="r"),
         "vy": np.load(VD / "labels.npy", mmap_mode="r"), "vok": np.load(VD / "ok.npy", mmap_mode="r")}
    z = np.load(VD / "index.npz")
    for r, tok in toks:
        out = builder.compute_targets(_scene(tok))
        assert out["ck_row"].dtype == torch.int64 and int(out["ck_row"]) == r
        for k, (shape, dt) in SHAPES.items():
            assert tuple(out[k].shape) == shape and out[k].dtype == dt, (k, out[k].shape, out[k].dtype)
        ref = gtl.load(tok)                                     # ref_* + kd_bev_0 / kd_ok_0 (DET) / kd_bev_1 (MAP)
        for k, v in ref.items():
            assert torch.equal(out[k], v), k
        assert out["kd_bev_0"].shape == (256, 50, 100) and out["kd_bev_0"].dtype == torch.float16
        assert bool(out["kd_ok_0"]) and bool(out["kd_ok_1"])
        assert bool(out["ck2_wu_ok"])
        np.testing.assert_array_equal(out["ck2_gt"].numpy(), A["gt"][r])
        np.testing.assert_array_equal(out["ck2_raw_y"].numpy(), ck_targets(np.asarray(A["ry"][r]), "official"))
        np.testing.assert_array_equal(out["ck2_raw_ok"].numpy(), A["rok"][r])
        np.testing.assert_array_equal(out["ck2_var_traj"].numpy(), A["vt"][r])
        np.testing.assert_array_equal(out["ck2_var_y"].numpy(), ck_targets(np.asarray(A["vy"][r]), "official"))
        # default 'official' = the old raw label columns; 'decoupled' differs only in the EP column
        w_dec = b_dec.warmup_row(r)
        for k in ("ck2_raw_y", "ck2_var_y"):
            src = np.asarray(A["ry" if k == "ck2_raw_y" else "vy"][r])
            np.testing.assert_array_equal(out[k].numpy(), src[:, CKI])
            np.testing.assert_array_equal(w_dec[k].numpy(), ck_targets(src, "decoupled"))
            np.testing.assert_array_equal(np.delete(w_dec[k].numpy(), 2, -1), np.delete(src[:, CKI], 2, -1))
        for k in ("ck2_raw_ok", "ck2_var_ok", "ck2_wu_ok", "ck2_gt", "ck2_var_traj", "ck2_var_valid"):
            assert torch.equal(w_dec[k], out[k]), k
        np.testing.assert_array_equal(out["ck2_var_ok"].numpy(), A["vok"][r])
        np.testing.assert_array_equal(out["ck2_var_valid"].numpy(), z["valid"][r].reshape(96))
        np.testing.assert_array_equal(out["ck2_var_anchor"].numpy(), z["anchor_idx"][r])


@need
def test_builder_missing_token_and_guards(builder):
    out = builder.compute_targets(_scene("ffffffffffffffff"))
    assert int(out["ck_row"]) == -1 and not bool(out["ck2_wu_ok"]) and not bool(out["ref_gt_ok"])
    for k, (shape, dt) in SHAPES.items():
        assert tuple(out[k].shape) == shape and out[k].dtype == dt, k
        assert float(out[k].float().abs().sum()) == 0.0, k
    for i in (0, 1):
        assert not bool(out[f"kd_ok_{i}"])
    with pytest.raises(ValueError):
        D2.CK2E2ETargetBuilder({"topk": 8})
    with pytest.raises(ValueError):
        D2.CK2E2ETargetBuilder({"teacher_det_run": ""})
    b = D2.CK2E2ETargetBuilder({"teacher_map_run": ""})
    assert b.teacher_runs == (D2.TEACHER_DET_RUN,) and b.ep_target == "official"
    with pytest.raises(ValueError, match="ep_target"):
        D2.CK2E2ETargetBuilder({"ep_target": "bogus"})
    # config forms: ParaSSRConfig-like namespace, nested warm-up override, DictConfig
    from omegaconf import OmegaConf
    ns = SimpleNamespace(ck_e2e2={"warmup": {"n_var": 16}}, ref_data_root="/x")
    b2 = D2.CK2E2ETargetBuilder(ns)
    assert b2.wcfg["n_var"] == 16 and b2.wcfg["var_dir"] == W["var_dir"] and b2.ref_data_root == "/x"
    b3 = D2.CK2E2ETargetBuilder(OmegaConf.create({"warmup": {"sampler_seed": 3}}))
    assert b3.wcfg["sampler_seed"] == 3


def test_builder_structural_error_raises(tmp_path):
    """A wrong-shaped warm-up source is a configuration error (raises); a missing row is not."""
    pk, rl, vd = tmp_path / "pk", tmp_path / "rl", tmp_path / "vd"
    for d in (pk, rl, vd):
        d.mkdir()
    n = 3
    pd.DataFrame({"token": [f"t{i:015d}" for i in range(n)], "log": ["L"] * n, "row": np.arange(n)}).to_parquet(
        pk / "tokens.parquet")
    np.save(pk / "ok.npy", np.ones(n, bool))
    np.save(pk / "gt_traj.npy", np.zeros((n, 8, 3), np.float32))
    np.save(rl / "labels.npy", np.zeros((n, 256, 9), np.float32))
    np.save(rl / "ok.npy", np.ones((n, 256), bool))
    np.save(vd / "traj.npy", np.zeros((n, 95, 8, 3), np.float32))                    # wrong K
    np.save(vd / "labels.npy", np.zeros((n, 96, 9), np.float32))
    np.save(vd / "ok.npy", np.ones((n, 96), bool))
    np.savez(vd / "index.npz", valid=np.ones((n, 16, 6), bool), anchor_idx=np.zeros((n, 16), np.int16),
             built=np.ones(n, bool))
    src = D2._WUArrays(D2.warmup_cfg({"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)}))
    with pytest.raises(D2.WUSourceError):
        D2.wu_row(src, 0)
    assert not D2.wu_row(src, -1)["ck2_wu_ok"] and not D2.wu_row(src, 99)["ck2_wu_ok"]


def _wu_fixture(root: Path, n: int = 4):
    """a structurally valid n-row warm-up source set whose build.json / packed meta / raw-label plan carry the true
    provenance (token_sha16 of the packed token list, anchors sha16, AnchorSampler(seed 0) config)."""
    import json

    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from tools.ck.data import common as CM
    pk, rl, vd = root / "pk", root / "rl", root / "vd"
    for d in (pk, rl / "shards", vd):
        d.mkdir(parents=True)
    toks = [f"t{i:015d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": ["L"] * n, "row": np.arange(n)}).to_parquet(pk / "tokens.parquet")
    sha = CM.sha16_array(np.array(toks, "U16"))
    np.save(pk / "ok.npy", np.ones(n, bool))
    np.save(pk / "gt_traj.npy", np.zeros((n, 8, 3), np.float32))
    np.save(rl / "labels.npy", np.zeros((n, 256, 9), np.float32))
    np.save(rl / "ok.npy", np.ones((n, 256), bool))
    np.save(vd / "traj.npy", np.zeros((n, 96, 8, 3), np.float32))
    np.save(vd / "labels.npy", np.zeros((n, 96, 9), np.float32))
    np.save(vd / "ok.npy", np.ones((n, 96), bool))
    np.savez(vd / "index.npz", valid=np.ones((n, 16, 6), bool), anchor_idx=np.zeros((n, 16), np.int16),
             built=np.ones(n, bool))
    smp = AS.AnchorSampler(seed=0).config()
    build = {"names": list(D2.VNAMES), "n_anchors": 16, "seed": 0, "sampler": smp, "anchors_sha16": AS.ANCHORS_SHA16,
             "n_tokens": n, "token_sha16": sha}
    (vd / "build.json").write_text(json.dumps(build))
    (pk / "meta.json").write_text(json.dumps({"token_sha16": sha}))
    (rl / "shards" / "_plan.json").write_text(json.dumps({"token_sha16": sha}))
    return D2.warmup_cfg({"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)}), build


def test_warmup_provenance_guard(tmp_path):
    """report 48 S56-5 (task 5b): _WUArrays.check also proves the rows are the same tokens as the variant build /
    packed meta / raw-label plan (token_sha16), the anchors sha16 and the AnchorSampler config of the build; a re-pack
    that keeps N but swaps two tokens, a changed sampler or anchor file is refused (WUSourceError; the launcher turns it
    into a LaunchError and records the provenance in teachers.json)."""
    import json
    from tools.ck.e2e2 import launch_util2 as L2
    wcfg, build = _wu_fixture(tmp_path / "ok")
    prov = D2._WUArrays(wcfg).check()["provenance"]
    assert prov["variant build.json"] == prov["token_sha16"] == build["token_sha16"] and prov["sampler"] == "equal"
    assert prov["packed meta.json"] == prov["raw_labels shards/_plan.json"] == build["token_sha16"]
    # two rows swapped in the packed token list (same N, same shapes): refused
    w2, _ = _wu_fixture(tmp_path / "swap")
    tp = Path(w2["packed"]) / "tokens.parquet"
    df = pd.read_parquet(tp)
    df.loc[[1, 2], "token"] = df.loc[[2, 1], "token"].to_numpy()
    df.to_parquet(tp)
    with pytest.raises(D2.WUSourceError, match="token_sha16"):
        D2._WUArrays(w2).check()
    for name, over in (("sampler", {"sampler": {**build["sampler"], "n_mid": 4}}), ("anchors", {"anchors_sha16": "0" * 16}),
                       ("ntok", {"n_tokens": 5})):
        w3, b3 = _wu_fixture(tmp_path / name)
        (Path(w3["var_dir"]) / "build.json").write_text(json.dumps({**b3, **over}))
        with pytest.raises(D2.WUSourceError):
            D2._WUArrays(w3).check()
    w4, _ = _wu_fixture(tmp_path / "pmeta")
    (Path(w4["packed"]) / "meta.json").write_text(json.dumps({"token_sha16": "f" * 16}))
    with pytest.raises(D2.WUSourceError, match="meta.json"):
        D2._WUArrays(w4).check()
    # launch: LaunchError on the swapped set, the real default sources pass and are recorded
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    with pytest.raises(L2.LaunchError, match="token_sha16"):
        L2.warmup_check(CKE2E2Config.from_any({"warmup": {k: w2[k] for k in ("packed", "raw_labels", "var_dir")}}))
    if HAVE:
        rec = L2.warmup_check(CKE2E2Config.from_any(None))
        assert rec["provenance"]["sampler"] == "equal" and rec["n_rows"] == D2.N_ROWS


class _DS(torch.utils.data.Dataset):
    def __init__(self, b, toks):
        self.b, self.toks = b, toks

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        return self.b.compute_targets(_scene(self.toks[i]))


@need
def test_builder_pickle_workers(toks):
    b = D2.CK2E2ETargetBuilder({"teacher_det_run": TDET, "teacher_map_run": TMAP})
    tl = [t for _, t in toks] + ["ffffffffffffffff"]
    ref = [b.compute_targets(_scene(t)) for t in tl]           # opens files in the main process
    assert b._gtl is not None and b.src._mm
    s = pickle.dumps(b)
    b2 = pickle.loads(s)
    assert b2._gtl is None and not b2.src._mm and b2.rows._pid is None
    assert len(s) < 100_000                                     # no memmaps / index arrays in the pickle
    for ctx in ("fork", "spawn"):
        dl = torch.utils.data.DataLoader(_DS(b, tl), batch_size=None, num_workers=2, multiprocessing_context=ctx)
        got = list(dl)
        for g, r in zip(got, ref):
            assert set(g) == set(r)
            for k in r:
                assert torch.equal(g[k], r[k]), (ctx, k)


# ----------------------------------------------------------------------------------------------- warm-up sampler
def _batch(builder, rows_toks):
    outs = [builder.compute_targets(_scene(t)) for _, t in rows_toks]
    return {k: torch.stack([o[k] for o in outs]) for k in outs[0] if k.startswith(("ck2_", "ck_row"))}


@need
@pytest.mark.parametrize("epoch", [0, 3])
def test_warmup_sampler_equals_ck2dataset(builder, ds, epoch):
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rows = [0, 40000, 12345, 85108]
    rt = [(r, str(tdf.token.iloc[r])) for r in rows]
    t = _batch(builder, rt)
    toks_ = [tk for _, tk in rt]
    ws = D2.WarmupSampler(sampler_seed=0, seed=0, n_var=32)
    S = ws.sample(toks_, {k: t[k] for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_var_valid")}, epoch)
    assert S["anc_idx"].shape == (4, 32) and S["group"].dtype == np.int8 and S["ok"].all()
    # the same draw with the CK2Dataset variant rule ('uniform', stream 'var') == CK2Dataset exactly (same EP target;
    # the anchor draw itself never sees EP: builder targets are decoupled, the teacher dataset samples on official)
    wu = D2.WarmupSampler(sampler_seed=0, seed=0, n_var=32, var_sampling="uniform")
    Su = wu.sample(toks_, {k: t[k] for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_var_valid")}, epoch, stream="var")
    for b, r in enumerate(rows):
        i = int(np.searchsorted(ds.rows, r))
        assert ds.rows[i] == r
        it = ds.labels_item(i, epoch=epoch)
        assert np.array_equal(S["anc_idx"][b], it["anc_idx"]) and np.array_equal(S["group"][b], it["anc_group"])
        assert np.array_equal(t["ck2_raw_y"][b].numpy()[S["anc_idx"][b]], it["y"])
        assert np.array_equal(t["ck2_raw_ok"][b].numpy()[S["anc_idx"][b]], it["y_ok"])
        assert np.array_equal(Su["var_cols"][b], it["var_col"]) and np.array_equal(Su["anc_idx"][b], it["anc_idx"])
        assert np.array_equal(t["ck2_var_traj"][b].numpy()[Su["var_cols"][b]], it["var_traj"])
        assert np.array_equal(t["ck2_var_y"][b].numpy()[Su["var_cols"][b]], it["var_y"])
        # type-balanced default: per (token, epoch) token_rng stream 'ck2e2e.wu'
        exp = D2.type_balanced_cols(t["ck2_var_valid"][b].numpy().reshape(16, 6), 32,
                                    token_rng(toks_[b], epoch, 0, "ck2e2e.wu"))
        assert np.array_equal(S["var_cols"][b], exp[0]) and np.array_equal(S["var_pad_ok"][b], exp[1])
    # rows without warm-up data -> ok False, canonical group pattern, zeros
    t2 = dict(t)
    t2["ck2_wu_ok"] = torch.tensor([True, False, True, False])
    S2 = ws.sample(toks_, t2, epoch)
    assert S2["ok"].tolist() == [True, False, True, False]
    assert (S2["anc_idx"][1] == 0).all() and not S2["var_pad_ok"][1].any()
    assert ((S2["group"][1] <= 1).sum()) == 16
    assert np.array_equal(S2["anc_idx"][0], S["anc_idx"][0])


@need
def test_warmup_prior_equals_ck2_label_prior(ds):
    from tools.ck.data import ck2_dataset as CD
    a = D2.ck2_warmup_prior(None, n_max=4000)
    b = CD.ck2_label_prior(ds, n_max=4000)
    assert a.shape == (5,) and ds.ep_target == "official"         # default since 2026-10-09 00:30 KST
    np.testing.assert_array_equal(a, b)
    # config forms
    np.testing.assert_array_equal(D2.ck2_warmup_prior({"warmup": dict(W)}, n_max=300),
                                  CD.ck2_label_prior(ds, n_max=300))
    # 'decoupled' == the decoupled teacher dataset; only the EP prior moves
    ds_dec = CD.CK2Dataset("navtrain_train", bev="none", n_var=32, var_name=VD.name, seed=0, sampler_seed=0,
                           ep_target="decoupled")
    p_dec = D2.ck2_warmup_prior({"ep_target": "decoupled"}, n_max=300)
    np.testing.assert_array_equal(p_dec, CD.ck2_label_prior(ds_dec, n_max=300))
    np.testing.assert_array_equal(p_dec, D2.ck2_warmup_prior(None, n_max=300, ep_target="decoupled"))
    p_off = D2.ck2_warmup_prior(None, n_max=300)
    np.testing.assert_array_equal(p_off, D2.ck2_warmup_prior({"ep_target": "official"}, n_max=300))
    np.testing.assert_array_equal(np.delete(p_off, 2), np.delete(p_dec, 2))
    assert p_dec[2] > p_off[2]                           # decoupled EP >= official EP everywhere


# ----------------------------------------------------------------------------------------------- warm-up candidates
@need
def test_warmup_cands_assembly(builder):
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rt = [(r, str(tdf.token.iloc[r])) for r in (0, 40000, 777)] + [(-1, "ffffffffffffffff")]
    t = _batch(builder, rt)
    toks_ = [tk for _, tk in rt]
    ws = D2.WarmupSampler(n_var=32)
    out = C2.warmup_cands(t, toks_, 2, ws)
    S = ws.sample(toks_, t, 2)
    anc = ws.anchors
    assert out["cand"].shape == (4, 64, 8, 3) and out["y"].shape == (4, 64, 5) and out["ok"].shape == (4, 64)
    assert out["wu_ok"].tolist() == [True, True, True, False]
    for b in range(3):
        ai, vc = S["anc_idx"][b], S["var_cols"][b]
        np.testing.assert_array_equal(out["cand"][b, :32].numpy(), anc[ai])
        np.testing.assert_array_equal(out["cand"][b, 32:].numpy(), t["ck2_var_traj"][b].numpy()[vc])
        np.testing.assert_array_equal(out["y"][b, :32].numpy(), t["ck2_raw_y"][b].numpy()[ai])
        np.testing.assert_array_equal(out["y"][b, 32:].numpy(), t["ck2_var_y"][b].numpy()[vc])
        exp_ok = np.concatenate([t["ck2_raw_ok"][b].numpy()[ai],
                                 t["ck2_var_ok"][b].numpy()[vc] & t["ck2_var_valid"][b].numpy()[vc]
                                 & S["var_pad_ok"][b]])
        np.testing.assert_array_equal(out["ok"][b].numpy(), exp_ok)
        g = out["group"][b].numpy()
        assert (g[32:] == 3).all() and np.array_equal(g[:32], S["group"][b])
        assert np.array_equal(out["vtype"][b, 32:].numpy(), vc % 6) and (out["vtype"][b, :32] == 0).all()
        sp = out["sur_pos"][b].numpy()
        assert len(sp) == 16 and (g[sp] <= 1).all() and np.all(np.diff(sp) > 0)      # 8 near + 8 mid, slot order
        assert (out["lat_mask"][b].numpy() == ((g <= 1) | (g == 3))).all()
        assert out["teacher_ok"][b, :32].all()
        par = out["parent_anchor"][b].numpy()
        assert np.array_equal(par[:32], ai)
        assert np.array_equal(par[32:], t["ck2_var_anchor"][b].numpy()[vc // 6])
    # NU44: a token without warm-up data has every CK candidate masked (but finite geometry)
    assert not out["ok"][3].any() and not out["teacher_ok"][3].any() and torch.isfinite(out["cand"][3]).all()
    assert len(out["sur_pos"][3]) == 16


@need
def test_fill_fallback(builder):
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rt = [(r, str(tdf.token.iloc[r])) for r in (5, 40000)] + [(-1, "ffffffffffffffff")]
    t = _batch(builder, rt)
    toks_ = [tk for _, tk in rt]
    store = D2.LabelStore2(Path("/nonexistent_ck2_io_dir"), n_rows=D2.N_ROWS, ep_target=builder.ep_target)
    store.refresh(4)                                                  # no generation at all
    G = store.lookup(torch.tensor([r for r, _ in rt]), toks_, epoch=5, seed=0)
    assert not G["has"].any() and G["G_traj"].shape == (3, 48, 8, 3)
    ws = D2.WarmupSampler(n_var=32)
    G = C2.fill_fallback(G, t, toks_, 5, ws)
    assert G["G_src"].tolist() == [D2.SRC_FALLBACK, D2.SRC_FALLBACK, D2.SRC_NODATA]
    assert G["G_epoch"].tolist() == [-1, -1, -1]
    S = ws.sample(toks_, t, 5, stream=D2.STREAM_LAB)
    Swu = ws.sample(toks_, t, 5)
    for b in range(2):
        a16 = S["anc_idx"][b][S["group"][b] <= 1]
        assert np.array_equal(a16, Swu["anc_idx"][b][Swu["group"][b] <= 1])   # = this epoch's near+mid draw
        np.testing.assert_array_equal(G["G_traj"][b, :16].numpy(), ws.anchors[a16])
        np.testing.assert_array_equal(G["G_traj"][b, 16:].numpy(), t["ck2_var_traj"][b].numpy()[S["var_cols"][b]])
        np.testing.assert_array_equal(G["G_y"][b, :16].numpy(), t["ck2_raw_y"][b].numpy()[a16])
        np.testing.assert_array_equal(G["G_y"][b, 16:].numpy(), t["ck2_var_y"][b].numpy()[S["var_cols"][b]])
        assert G["G_ok"][b].all()
        assert (G["G_vtype"][b, :16] == 0).all() and np.array_equal(G["G_vtype"][b, 16:].numpy(),
                                                                     S["var_cols"][b] % 6)
        assert not np.array_equal(S["var_cols"][b], Swu["var_cols"][b])         # stream 'ck2e2e.lab' != 'ck2e2e.wu'
    assert not G["G_ok"][2].any() and float(G["G_traj"][2].abs().sum()) == 0.0
    # the combined helper
    G2 = C2.onpolicy_label_set(store, t, torch.tensor([r for r, _ in rt]), toks_, 5, 0, ws)
    for k in G:
        assert torch.equal(G[k], G2[k]), k


# ----------------------------------------------------------------------------------------------- fixture (NU40)
def make_ck2_fixture(path: Path = FIX2, src: Path = FIX1) -> Path:
    """CK1 real-batch fixture (read-only) -> CK2 fixture: same features and v2 targets, CK1 target keys replaced by
    CK2E2ETargetBuilder output (default ck2T / ck2M teacher runs)."""
    d = torch.load(src, map_location="cpu", weights_only=False)
    b = D2.CK2E2ETargetBuilder({"teacher_det_run": TDET, "teacher_map_run": TMAP})
    outs = [b.compute_for_token(t) for t in d["tokens"]]
    new = {k: torch.stack([o[k] for o in outs]) for k in outs[0]}
    tg = {k: v for k, v in d["targets"].items() if not k.startswith(CK1_TARGET_PREFIX)}
    tg.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save({"tokens": d["tokens"], "rows": d["rows"], "features": d["features"], "targets": tg,
                "source": str(src), "teacher_runs": list(b.teacher_runs), "warmup": b.wcfg}, tmp)
    os.replace(tmp, path)
    return path


@need
@pytest.mark.skipif(not FIX1.is_file(), reason="CK1 fixture absent")
def test_ck2_fixture():
    if not FIX2.is_file():
        make_ck2_fixture()
    d1 = torch.load(FIX1, map_location="cpu", weights_only=False)
    d2 = torch.load(FIX2, map_location="cpu", weights_only=False)
    assert d2["tokens"] == d1["tokens"] and d2["rows"] == d1["rows"] == [0, 40000]
    for k, v in d1["features"].items():
        assert torch.equal(d2["features"][k], v), k
    t1, t2 = d1["targets"], d2["targets"]
    assert not any(k.startswith("ck_p1_") for k in t2)
    for k, v in t1.items():
        if not k.startswith(CK1_TARGET_PREFIX):
            assert torch.equal(t2[k], v), k                         # v2 targets untouched
        elif not k.startswith("ck_p1_"):
            assert torch.equal(t2[k], v), k                         # ck_row / ref_* / kd_*: same caches as CK1
    for k, (shape, dt) in SHAPES.items():
        assert t2[k].shape == (2,) + shape and t2[k].dtype == dt, k
    assert t2["ck2_wu_ok"].all() and t2["kd_ok_0"].all() and t2["kd_ok_1"].all()
    # sampler GT == v2 target trajectory on the fixture rows (anchor_sampler doc [실측])
    assert torch.equal(t2["ck2_gt"], t2["trajectory"].float())
    st = np.asarray(np.load(PK / "status.npy", mmap_mode="r")[[0, 40000]])
    assert np.array_equal(d2["features"]["status_feature"].numpy(), st)


def test_misc_helpers(tmp_path):
    assert D2.cfg2_dict(None) == {} and D2.cfg2_dict({"a": 1}) == {"a": 1}
    assert D2.cfg2_dict(SimpleNamespace(ck_e2e2={"b": 2})) == {"b": 2}
    w = D2.warmup_cfg({"var_dir": "/x"})
    assert w["var_dir"] == "/x" and w["packed"] == W["packed"]
    w2 = D2.warmup_cfg({"warmup": {"n_var": 8}})
    assert w2["n_var"] == 8
    with pytest.raises(ValueError):
        D2.WarmupSampler(var_sampling="bogus")
