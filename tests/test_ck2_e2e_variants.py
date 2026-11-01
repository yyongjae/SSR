"""CK2 e2e T5: online variants (navsim/agents/para_ssr/ck/cands2.online_variants, ext 'straight') == the warm-up variant
label files (var_separate_sampler16_accstraight: traj.npy, index.npz valid) on real navtrain_train rows; on-policy
candidate assembly (onpolicy_cands).  CPU only.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_variants.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import anchor_sampler as AS
from navsim.agents.para_ssr.ck import cands2 as C2
from navsim.agents.para_ssr.ck import e2e_data2 as D2
from navsim.agents.para_ssr.ck.anchor_sampler import token_rng

VD = Path(D2.WARMUP_DEFAULT["var_dir"])
PK = Path(D2.WARMUP_DEFAULT["packed"])
HAVE = (VD / "traj.npy").is_file() and (VD / "index.npz").is_file() and (PK / "status.npy").is_file()
need = pytest.mark.skipif(not HAVE, reason="variant label files / packed status absent")
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def real():
    z = np.load(VD / "index.npz")
    valid = z["valid"]
    inval = np.flatnonzero(~valid.all((1, 2)))
    rows = np.unique(np.concatenate([[0, 40000, len(valid) - 1], inval[:6],
                                     np.random.default_rng(0).choice(len(valid), 7, replace=False)]))
    st = np.asarray(np.load(PK / "status.npy", mmap_mode="r")[rows])
    anc = AS.load_anchors()
    return dict(rows=rows, cand=torch.from_numpy(anc[z["anchor_idx"][rows].astype(np.int64)]),
                status=torch.from_numpy(st), valid=valid[rows].reshape(len(rows), 96),
                traj=np.asarray(np.load(VD / "traj.npy", mmap_mode="r")[rows]), n_inval=len(inval))


@need
def test_online_variants_equal_label_files(real):
    assert real["n_inval"] > 0 and not real["valid"].all()           # rows with lateral duplicates are covered
    V = C2.online_variants(real["cand"], C2.v0_from_status(real["status"]))
    t = V["traj96"].numpy()
    assert t.dtype == np.float32 and t.shape == (len(real["rows"]), 96, 8, 3)
    assert np.array_equal(t[:, ::6].view(np.uint32), real["cand"].numpy().view(np.uint32))     # identity bitwise
    d = np.abs(t - real["traj"])
    assert d.max() <= 1e-5, d.max()
    # [실측 2026-10-08] bitwise equal on CPU with the float64 v0 (same make_variants code path as label_variants)
    assert np.array_equal(t.view(np.uint32), real["traj"].view(np.uint32))
    assert np.array_equal(V["valid96"].numpy(), real["valid"])
    assert torch.equal(V["vtype"], torch.arange(96) % 6) and torch.equal(V["parent"], torch.arange(96) // 6)
    assert V["dev_xy96"].shape == (len(real["rows"]), 96) and float(V["dev_xy96"][:, ::6].abs().max()) == 0.0
    assert V["fin16"].all()


@need
def test_online_variants_f32_v0_close(real):
    """v0 from the f32 status hypot (= refiner.e2e.ego_inputs) instead of the f64 one: <= 1e-5."""
    s = real["status"].float()
    V = C2.online_variants(real["cand"], torch.hypot(s[:, 4], s[:, 5]))
    assert np.abs(V["traj96"].numpy() - real["traj"]).max() <= 1e-5


def test_online_variants_nonfinite_parent_and_config():
    anc = torch.from_numpy(AS.load_anchors()[:32].reshape(2, 16, 8, 3).copy())
    v0 = torch.tensor([3.0, 7.0], dtype=torch.float64)
    ref = C2.online_variants(anc, v0)
    bad = anc.clone()
    bad[0, 3, 2, 1] = float("nan")
    V = C2.online_variants(bad, v0)
    assert torch.isnan(V["traj96"][0, 18:24]).all() and not V["valid96"][0, 18:24].any()
    assert not bool(V["fin16"][0, 3]) and bool(V["fin16"][0, 2])
    keep = torch.ones(96, dtype=torch.bool)
    keep[18:24] = False
    assert torch.equal(V["traj96"][0, keep], ref["traj96"][0, keep]) and torch.equal(V["traj96"][1], ref["traj96"][1])
    # const_curv changes only the a+0.5 column (mode B beyond S_8); the other 5 columns are ext-independent
    Vc = C2.online_variants(anc, v0, {"ext": "const_curv"})
    other = (torch.arange(96) % 6) != 3
    assert torch.equal(Vc["traj96"][:, other], ref["traj96"][:, other])
    with pytest.raises(ValueError):
        C2.variants_cfg({"combine": "cross"})                     # 12 columns != generation layout
    with pytest.raises(ValueError):
        C2.variants_cfg({"ext": "centerline_gt"})
    with pytest.raises(ValueError):
        C2.variants_cfg({"bogus": 1})
    with pytest.raises(ValueError):
        C2.online_variants(anc[0], v0)


@need
def test_check_variant_files():
    got = C2.check_variant_files(VD)
    assert got["ext"] == "straight" and got["names"] == list(D2.VNAMES)
    with pytest.raises(ValueError):
        C2.check_variant_files(VD, {"ext": "const_curv"})
    with pytest.raises(ValueError):
        C2.check_variant_files(VD, {"s_on_frac": 0.3})


def test_onpolicy_cands_assembly():
    g = torch.Generator().manual_seed(0)
    anc = torch.from_numpy(AS.load_anchors())
    idx = torch.randint(0, 256, (3, 16), generator=g)
    top = anc[idx] + 0.01 * torch.randn(3, 16, 8, 3, generator=g)
    v0 = torch.tensor([0.5, 4.0, 9.0], dtype=torch.float64)
    toks = ["t" * 16, "u" * 16, "v" * 16]
    out = C2.onpolicy_cands(top, v0, toks, epoch=7, seed=0, n_var=32)
    V = out["V"]
    assert out["cand"].shape == (3, 48, 8, 3) and out["ok"].shape == (3, 48) and out["vtype"].shape == (3, 48)
    assert torch.equal(out["cand"][:, :16], top.float())
    cols, pad = D2.sample_cols(V["valid96"].numpy(), toks, 7, 0, D2.STREAM_NOW, 32)
    assert np.array_equal(out["cols"].numpy(), cols) and np.array_equal(out["pad_ok"].numpy(), pad)
    for b in range(3):
        assert torch.equal(out["cand"][b, 16:], V["traj96"][b, out["cols"][b]])
        exp_ok = V["valid96"][b, out["cols"][b]] & out["pad_ok"][b]
        assert torch.equal(out["ok"][b, 16:], exp_ok) and out["ok"][b, :16].all()
        assert torch.equal(out["vtype"][b, 16:], out["cols"][b] % 6) and (out["vtype"][b, 16:] > 0).all()
        assert torch.equal(out["parent"][b, 16:], out["cols"][b] // 6)
        assert torch.equal(out["parent"][b, :16], torch.arange(16))
    # same token / epoch -> same draw; another epoch -> another draw; V reuse gives identical output
    again = C2.onpolicy_cands(top, v0, toks, 7, 0, V=V)
    assert torch.equal(again["cols"], out["cols"]) and torch.equal(again["cand"], out["cand"])
    assert not torch.equal(C2.onpolicy_cands(top, v0, toks, 8, 0, V=V)["cols"], out["cols"])
    # recorded 96 for the labeler: the exact tensors used in the step
    assert torch.equal(V["traj96"][:, ::6], top.float())
    # non-finite parent: its own and its variants' columns are masked, the rest untouched
    bad = top.clone()
    bad[1, 0, 0, 0] = float("inf")
    ob = C2.onpolicy_cands(bad, v0, toks, 7, 0)
    assert not bool(ob["ok"][1, 0]) and torch.isfinite(ob["cand"]).all()
    assert not ob["ok"][1, 16:][ob["parent"][1, 16:] == 0].any()
    with pytest.raises(ValueError):
        C2.onpolicy_cands(top[:, :8], v0, toks, 7, 0)


def test_onpolicy_token_rng_stream_is_now():
    anc = torch.from_numpy(AS.load_anchors()[:16])[None]
    V = C2.online_variants(anc, torch.tensor([5.0], dtype=torch.float64))
    out = C2.onpolicy_cands(anc, None, ["abcdabcdabcdabcd"], 9, 3, V=V)
    exp = D2.type_balanced_cols(V["valid96"][0].numpy().reshape(16, 6), 32,
                                token_rng("abcdabcdabcdabcd", 9, 3, "ck2e2e.now"))
    assert np.array_equal(out["cols"][0].numpy(), exp[0])
