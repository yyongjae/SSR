"""CK2 variant labelling (tools/ck/data/label_variants.py): build layout / subsets / resume guard on a tiny fake root
made of the first 5 real navtrain_train packed rows, and (CK2_SCORER_TEST=1) the official scoring end to end with the
identity rows checked against raw256.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_label_variants.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.ck.data import label_variants as LV  # noqa: E402
from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck import variants as VV  # noqa: E402

CM = LV.CM
SPLIT = "navtrain_train"
N_FAKE = 5


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    real = CM.packed_dir(SPLIT)
    raw = CM.labels_dir(SPLIT, "raw256")
    need = [real / f for f in ("tokens.parquet", "gt_traj.npy", "status.npy")] + [raw / "labels.npy", raw / "ok.npy"]
    if not all(p.is_file() for p in need) or not Path(str(AS.OFFICIAL_LABELS) + ".npy").is_file():
        pytest.skip("real packed / raw256 data absent")
    root = tmp_path_factory.mktemp("ck2root")
    pk = CM.packed_dir(SPLIT, root)
    pk.mkdir(parents=True)
    pd.read_parquet(real / "tokens.parquet").iloc[:N_FAKE].to_parquet(pk / "tokens.parquet")
    for f in ("gt_traj.npy", "status.npy"):
        np.save(pk / f, np.asarray(np.load(real / f, mmap_mode="r")[:N_FAKE]))
    rd = CM.labels_dir(SPLIT, "raw256", root)
    rd.mkdir(parents=True)
    for f in ("labels.npy", "ok.npy"):
        np.save(rd / f, np.asarray(np.load(raw / f, mmap_mode="r")[:N_FAKE]))
    return root


def _packed(root):
    pk = CM.packed_dir(SPLIT, root)
    toks = pd.read_parquet(pk / "tokens.parquet").token.astype(str).tolist()
    return toks, np.load(pk / "gt_traj.npy"), np.load(pk / "status.npy").astype(np.float64)


def test_config_validation(fake_root):
    with pytest.raises(ValueError):
        LV.make_config(SPLIT, subset="sampler", n_anchors=12, root=fake_root)
    with pytest.raises(ValueError):
        LV.make_config(SPLIT, subset="file", root=fake_root)
    with pytest.raises(ValueError):
        LV.make_config(SPLIT, combine="both", root=fake_root)
    a = LV.make_config(SPLIT, root=fake_root)
    b = LV.make_config(SPLIT, root=fake_root, limit=3)
    c = LV.make_config(SPLIT, root=fake_root, combine="cross")
    assert a["cfg_sha16"] == b["cfg_sha16"] != c["cfg_sha16"]           # limit is not part of the config hash
    assert a["n_variants"] == 6 and c["n_variants"] == 12 and c["names"][:6] == a["names"]


def test_build_sampler_cross_layout(fake_root, tmp_path):
    cfg = LV.make_config(SPLIT, combine="cross", subset="sampler", n_anchors=16, root=fake_root, limit=3)
    out = LV.out_dir(SPLIT, "t", tmp_path)
    info = LV.ensure_built(cfg, out, fake_root, 2, False)
    K, V = 16, 12
    assert info["n_built"] == 3
    traj = np.load(out / "traj.npy")
    ix = np.load(out / "index.npz")
    assert traj.shape == (N_FAKE, K * V, 8, 3) and traj.dtype == np.float32
    assert (traj[3:] == 0).all() and ix["built"].tolist() == [True] * 3 + [False] * 2
    assert (ix["anchor_idx"][3:] == -1).all()
    assert np.array_equal(ix["col_k"], np.repeat(np.arange(K), V)) and np.array_equal(ix["col_v"], np.tile(np.arange(V), K))
    anchors = AS.load_anchors()
    toks, gt, st = _packed(fake_root)
    smp = AS.AnchorSampler(label_source=AS.LabelSource(), seed=0)
    for r in range(3):
        s = smp.sample(toks[r], gt[r], epoch=0, shuffle=False)
        idx = ix["anchor_idx"][r].astype(np.int64)
        assert np.array_equal(idx, s["idx"][:16])                          # near + mid of epoch 0
        assert np.array_equal(ix["group"][r], s["group"][:16]) and np.allclose(ix["pi"][r], s["pi"][:16])
        assert np.array_equal(ix["pass"][r], s["pass_mask"][:16])
        t = traj[r].reshape(K, V, 8, 3)
        assert np.array_equal(t[:, 0], anchors[idx])                        # identity = anchor bytes
        v0 = float(np.nan_to_num(np.hypot(st[r, 4], st[r, 5]), nan=0.0))
        ref = VV.make_variants_np(anchors[idx], v0, combine="cross")
        assert np.array_equal(t, ref["traj"]) and np.array_equal(ix["valid"][r], ref["valid"])
        assert ix["v0"][r] == np.float32(v0)
    # resume: same config -> no rebuild; different config with shards present -> refuse
    m0 = (out / "traj.npy").stat().st_mtime_ns
    LV.ensure_built(cfg, out, fake_root, 1, False)
    assert (out / "traj.npy").stat().st_mtime_ns == m0
    (out / "shards").mkdir()
    with pytest.raises(SystemExit):
        LV.ensure_built(LV.make_config(SPLIT, combine="separate", root=fake_root, limit=3), out, fake_root, 1, False)


def test_build_nearest(fake_root, tmp_path):
    cfg = LV.make_config(SPLIT, combine="separate", subset="nearest", n_anchors=10, root=fake_root)
    out = LV.out_dir(SPLIT, "n", tmp_path)
    LV.ensure_built(cfg, out, fake_root, 1, False)
    ix = np.load(out / "index.npz")
    toks, gt, _ = _packed(fake_root)
    anchors = AS.load_anchors()
    for r in range(N_FAKE):
        d = AS.gt_distance(anchors, gt[r])
        assert np.array_equal(ix["anchor_idx"][r], np.argsort(d, kind="stable")[:10])
        assert np.array_equal(ix["rank"][r], np.arange(10))
    assert np.load(out / "traj.npy").shape == (N_FAKE, 60, 8, 3)


def test_build_idx_file(fake_root, tmp_path):
    idx = np.stack([np.arange(r, r + 4) for r in range(N_FAKE)]).astype(np.int64)
    f = tmp_path / "idx.npy"
    np.save(f, idx)
    cfg = LV.make_config(SPLIT, subset="file", n_anchors=4, idx_file=str(f), root=fake_root)
    out = LV.out_dir(SPLIT, "f", tmp_path)
    LV.ensure_built(cfg, out, fake_root, 1, False)
    ix = np.load(out / "index.npz")
    assert np.array_equal(ix["anchor_idx"], idx)
    t = np.load(out / "traj.npy").reshape(N_FAKE, 4, 6, 8, 3)
    assert np.array_equal(t[:, :, 0], AS.load_anchors()[idx])


@pytest.mark.skipif(os.environ.get("CK2_SCORER_TEST") != "1", reason="official scoring (set CK2_SCORER_TEST=1)")
def test_score_identity_equals_raw256(fake_root, tmp_path):
    res = LV.run(SPLIT, combine="cross", subset="sampler", n_anchors=16, name="s", out_root=tmp_path, root=fake_root,
                 limit=2, workers=2, chunk=1, build_workers=1)
    assert res["n_ok"] == res["n_expected"] == 2 * 16 * 12
    assert res["identity_vs_raw256"]["all_equal"] and res["identity_vs_raw256"]["n_compared"] == 32
    assert res["identity_vs_wote_pdm_score_256"]["all_equal"]
    assert res["invalid_vs_parent"]["all_equal"] in (True, None)
    out = LV.out_dir(SPLIT, "s", tmp_path)
    assert json.loads((out / "meta.json").read_text())["k"] == 192
    lab = np.load(out / "labels.npy")
    assert lab.shape == (N_FAKE, 192, 9) and np.isnan(lab[2:]).all()
