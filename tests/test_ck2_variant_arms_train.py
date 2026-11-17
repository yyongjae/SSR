"""Variant-set ablation arms, training path (group A-train): ck_e2e2.variants = base / v8 / v4 / spd4 / lat4
(ck/variants.VARIANT_ARMS) through cands2 (variants_cfg / layout_cfg / check_variant_files / warmup_cands /
onpolicy_cands / fill_fallback), e2e_data2 (column draws, warm-up files, WarmupSampler, rec2 chunks / CandRecorder2,
generations, LabelStore2, prior), online2 (validate bounds, CKE2E2 telemetry / KD-ok fraction / recorder / infer) and
tools/ck/e2e2/labeler2.py.  Base identity: frozen copies of the pre-layout code / constants below must equal the
default (base) path bit for bit.  CPU only; real data only where present (navtrain_val / navtrain_train arm label dirs).

  cd /workspace/SSR && set -a && . tools/release/paths.env && set +a && PYTHONDONTWRITEBYTECODE=1 \
    /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_variant_arms_train.py
"""
from __future__ import annotations

import json
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
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck.anchor_sampler import token_rng  # noqa: E402
from navsim.agents.para_ssr.ck.variants import (BASE_LAYOUT, BASE_VAR_NAME, VARIANT_ARMS,  # noqa: E402
                                                arm_layout, arm_var_name)

torch.set_num_threads(2)
ARMS = ("base", "v8", "v4", "spd4", "lat4")
NV_OF = {"base": 6, "v8": 9, "v4": 5, "spd4": 5, "lat4": 5}


def vcfg(arm):
    a = VARIANT_ARMS[arm]
    return {"speeds": list(a["speeds"]), "lats": list(a["lats"])}


def lay(arm):
    return arm_layout(arm)


# ----------------------------------------------------------------------------------------------- frozen base code
# verbatim copies of e2e_data2 at v2_final b902937 (NV = 6, N_TYPES = 5): the base path must equal them bit for bit
def _old_type_pools(valid):
    v = np.asarray(valid, bool).reshape(-1, 6)
    return [np.flatnonzero(v[:, t]) * 6 + t for t in range(1, 6)]


def _old_type_balanced_cols(valid, n, rng):
    n = int(n)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    pools = _old_type_pools(valid)
    quota = np.full(5, n // 5, np.int64)
    extra = n % 5
    if extra:
        quota[rng.permutation(5)[:extra]] += 1
    taken, rest = [], []
    for t, pool in enumerate(pools):
        p = pool[rng.permutation(len(pool))]
        taken.append(p[:quota[t]])
        rest.append(p[quota[t]:])
    cols = np.concatenate(taken).astype(np.int64)
    short = n - len(cols)
    rest_all = np.concatenate(rest).astype(np.int64)
    if short > 0 and len(rest_all):
        cols = np.concatenate([cols, rest_all[rng.permutation(len(rest_all))[:short]]])
    nv = len(cols)
    if nv == 0:
        return np.zeros(n, np.int64), np.zeros(n, bool)
    ok = np.ones(n, bool)
    if nv < n:
        cols = np.resize(cols, n)
        ok = np.arange(n) < nv
    perm = rng.permutation(n)
    return cols[perm].astype(np.int64), ok[perm]


def _old_uniform_cols(valid, n, rng):
    n = int(n)
    v = np.asarray(valid, bool).reshape(-1, 6).copy()
    v[:, 0] = False
    cols = np.flatnonzero(v.reshape(-1))
    cols = cols[rng.permutation(len(cols))]
    nv = len(cols)
    if nv >= n:
        return cols[:n].astype(np.int64), np.ones(n, bool)
    if nv == 0:
        return np.zeros(n, np.int64), np.zeros(n, bool)
    return np.resize(cols, n).astype(np.int64), np.arange(n) < nv


OLD_WU_SHAPES = {"p_ok": (), "gt": (8, 3), "raw_y": (256, 9), "raw_ok": (256,), "v_traj": (96, 8, 3),
                 "v_y": (96, 9), "v_ok": (96,), "valid": (16, 6), "anchor_idx": (16,), "built": ()}
OLD_GEN2_FILES = {"traj": ((96, 8, 3), np.float32), "labels": ((96, 9), np.float32), "cand_ok": ((96,), np.bool_),
                  "valid": ((96,), np.bool_), "row_state": ((), np.uint8)}
OLD_GEN2_ORDER = ("c = k*6 + v; k = v2 top-16 rank (plan_final_rewards desc); "
                  "v = id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5")
OLD_REC2_KEYS = {"row", "traj", "valid", "cand_idx", "v2_final", "v0", "gstep", "epoch", "rank", "fmt"}


def test_base_constants_unchanged():
    assert (D2.K16, D2.NV, D2.G_K, D2.N_TYPES) == (16, 6, 96, 5)
    assert D2.VNAMES == BASE_LAYOUT.names == ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")
    assert D2._WUArrays.SHAPES == OLD_WU_SHAPES and D2._wu_shapes(6) == OLD_WU_SHAPES
    assert {k: (s, t) for k, (s, t, _) in D2.GEN2_FILES.items()} == OLD_GEN2_FILES
    assert D2.GEN2_FORMAT == "ck2_96" and D2.gen2_format() == D2.gen2_format(BASE_LAYOUT) == "ck2_96"
    assert D2.GEN2_ORDER == OLD_GEN2_ORDER == D2.gen2_order(None)
    assert C2.VARIANTS_DEFAULT == O._d_variants() and C2.layout_cfg(None) == BASE_LAYOUT
    assert C2.variants_cfg(None) == C2.VARIANTS_DEFAULT                   # same keys / values as before
    for k, v in D2.empty_wu_row().items():
        assert v.shape == {"ck2_var_traj": (96, 8, 3), "ck2_var_y": (96, 5), "ck2_var_ok": (96,),
                           "ck2_var_valid": (96,)}.get(k, v.shape)


# ----------------------------------------------------------------------------------------------- layouts / config
def test_layout_cfg_per_arm():
    for arm in ARMS:
        L = C2.layout_cfg(vcfg(arm))
        assert L == lay(arm) and L.nv == NV_OF[arm] and L.g() == 16 * NV_OF[arm]
        assert L.is_base == (arm == "base")
        c = C2.variants_cfg(vcfg(arm))
        assert sorted(c) == sorted(C2.VARIANTS_DEFAULT)                    # no new key (C1)
        assert D2.layout_of({"variants": vcfg(arm)}) == L == D2.layout_of(vcfg(arm))
        assert D2.layout_names(L) == L.names
    assert lay("v8").names == ("id", "a-1.0", "a-0.5", "a+0.5", "a+1.0", "l-1.0", "l-0.5", "l+0.5", "l+1.0")
    assert lay("lat4").names == ("id", "l-1.0", "l-0.5", "l+0.5", "l+1.0") and lay("spd4").lats == ()
    # config forms: string lists (CLI), DictConfig, ParaSSRConfig-like, CKE2E2Config, empty list kept empty
    from omegaconf import OmegaConf
    assert C2.layout_cfg({"speeds": "[-0.5, 0.5]", "lats": "[-0.5,0.5]"}) == lay("v4")
    assert C2.layout_cfg({"speeds": [], "lats": [-1.0, -0.5, 0.5, 1.0]}) == lay("lat4")
    assert D2.layout_of(OmegaConf.create({"variants": vcfg("spd4")})) == lay("spd4")
    assert D2.layout_of(SimpleNamespace(ck_e2e2={"variants": vcfg("v8")})) == lay("v8")
    assert D2.layout_of(O.CKE2E2Config.from_any({"variants": vcfg("lat4")})) == lay("lat4")
    assert D2.layout_of(None) == D2.layout_of({}) == D2.layout_of({"warmup": {}}) == BASE_LAYOUT
    for bad in ({"combine": "cross"}, {"speeds": [], "lats": []}, {"speeds": [0.0]}, {"lats": [9.0]},
                {"speeds": [-0.5, -0.5]}, {"bogus": 1}, {"ext": "centerline_gt"}):
        with pytest.raises(ValueError):
            C2.variants_cfg(bad)
    with pytest.raises(ValueError):
        D2.layout_names(("a-1.0", "id"))
    with pytest.raises(ValueError):
        D2.layout_names("id")


def _validate_cfg(tmp_path, arm, **over):
    d = {"enabled": True, "io_dir": str(tmp_path / "io"), "use_teachers": False, "lambda_kd_score": 0.0,
         "lat_kd": {"balance": "fixed", "fixed": 0.0}, "kd_calib": {"enabled": False}, "score_prior": "none",
         "variants": vcfg(arm), "warmup": {"var_dir": str(tmp_path / "no_var_dir")}}
    d.update(over)
    return O.CKE2E2Config.from_any(d)


@pytest.mark.parametrize("arm", ARMS)
def test_validate_n_var_bounds(tmp_path, arm):
    n_max = 16 * (NV_OF[arm] - 1)                       # base 80, v8 128, v4 / spd4 / lat4 64
    c = _validate_cfg(tmp_path, arm)
    c.validate()
    assert c.to_dict()["variants"]["speeds"] == [float(a) for a in VARIANT_ARMS[arm]["speeds"]]
    for key, over in (("n_var_step", {}), ("warmup.n_var", "warmup"), ("onpolicy_label.n_var", "onpolicy_label")):
        for n, ok in ((n_max, True), (n_max + 1, False)):
            if over == {}:
                kw = {"n_var_step": n}
            elif over == "warmup":
                kw = {"warmup": {"var_dir": str(tmp_path / "no_var_dir"), "n_var": n}}
            else:
                kw = {"onpolicy_label": {"n_var": n}}
            cc = _validate_cfg(tmp_path, arm, **kw)
            if ok:
                cc.validate()
            else:
                with pytest.raises(ValueError, match=key.replace(".", r"\.") + rf" must be in \[0, {n_max}\]"):
                    cc.validate()
    assert O.CKE2E2(c).layout == lay(arm)


def _train_dir(arm):
    return Path(D2.WARMUP_DEFAULT["var_dir"]).parent / arm_var_name(arm)


def test_validate_refuses_label_files_of_another_arm(tmp_path):
    base_dir = Path(D2.WARMUP_DEFAULT["var_dir"])
    assert base_dir.name == BASE_VAR_NAME
    if not (base_dir / "build.json").is_file():
        pytest.skip("base warm-up label dir absent")
    for arm in ("v8", "v4", "spd4", "lat4"):
        with pytest.raises(ValueError, match="label files"):
            _validate_cfg(tmp_path, arm, warmup={"var_dir": str(base_dir)}).validate()
    for arm in ("v8", "v4"):
        d = _train_dir(arm)
        if (d / "build.json").is_file():
            _validate_cfg(tmp_path, arm, warmup={"var_dir": str(d)}).validate()
            with pytest.raises(ValueError, match="label files"):
                _validate_cfg(tmp_path, "base", warmup={"var_dir": str(d)}).validate()
            got = C2.check_variant_files(d, vcfg(arm))
            assert got["names"] == list(lay(arm).names) and got["n_variants"] == lay(arm).nv
            assert got["index_names"] == list(lay(arm).names)
    # v4 / spd4 / lat4 are all 16 x 5 = 80 wide: the names tell them apart
    d4 = _train_dir("v4")
    if (d4 / "build.json").is_file():
        for other in ("spd4", "lat4"):
            with pytest.raises(ValueError, match="label files"):
                _validate_cfg(tmp_path, other, warmup={"var_dir": str(d4)}).validate()


# ----------------------------------------------------------------------------------------------- column draws
def test_draws_base_identity_frozen():
    g = np.random.default_rng(0)
    for i in range(150):
        v = g.random((16, 6)) > g.choice([0.02, 0.3, 0.7, 0.97])
        for n in (0, 1, 7, 32, 33, 80, 95):
            for shape in ((16, 6), (96,)):
                vv = v.reshape(shape)
                a = D2.type_balanced_cols(vv, n, np.random.default_rng(i * 97 + n))
                b = _old_type_balanced_cols(vv, n, np.random.default_rng(i * 97 + n))
                assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)), (i, n, shape)
                a = D2.uniform_cols(vv, n, np.random.default_rng(i + n))
                b = _old_uniform_cols(vv, n, np.random.default_rng(i + n))
                assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)), (i, n, shape)
            # nv = 6 given explicitly == the default
            a = D2.draw_cols(v, n, np.random.default_rng(i), "type_balanced", nv=6)
            b = _old_type_balanced_cols(v, n, np.random.default_rng(i))
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
    valid = g.random((5, 96)) > 0.2
    toks = [f"{c}" * 16 for c in "abcde"]
    c1, o1 = D2.sample_cols(valid, toks, 4, 1, D2.STREAM_NOW, 32)
    for b, t in enumerate(toks):
        e = _old_type_balanced_cols(valid[b], 32, token_rng(t, 4, 1, D2.STREAM_NOW))
        assert np.array_equal(c1[b], e[0]) and np.array_equal(o1[b], e[1])


@pytest.mark.parametrize("arm", ("v8", "v4", "spd4", "lat4"))
def test_draws_per_arm(arm):
    L = lay(arm)
    nv, g_k = L.nv, L.g()
    full = np.ones((16, nv), bool)
    for s in range(10):
        for mode in ("type_balanced", "uniform"):
            cols, ok = D2.draw_cols(full, 32, np.random.default_rng(s), mode)
            assert ok.all() and len(np.unique(cols)) == 32 and cols.max() < g_k and (cols % nv != 0).all()
        cols, _ = D2.type_balanced_cols(full, 32, np.random.default_rng(s))
        assert np.bincount(cols % nv, minlength=nv).tolist() == [0] + [32 // (nv - 1)] * (nv - 1)   # v8 4 / V=5 8
    cols, ok = D2.type_balanced_cols(full, 16 * (nv - 1), np.random.default_rng(1))
    assert ok.all() and sorted(cols.tolist()) == [c for c in range(g_k) if c % nv]
    # flat valid: ambiguous width without nv (v4 / spd4 / lat4 are all 80), explicit nv == the 2-D call
    with pytest.raises(ValueError, match="pass nv"):
        D2.type_balanced_cols(full.reshape(-1), 32, np.random.default_rng(0))
    v = np.random.default_rng(3).random((16, nv)) > 0.3
    a = D2.type_balanced_cols(v, 32, np.random.default_rng(5))
    b = D2.type_balanced_cols(v.reshape(-1), 32, np.random.default_rng(5), nv=nv)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        D2.type_balanced_cols(v, 32, np.random.default_rng(5), nv=6)          # [16, V] is not [*, 6]
    toks = ["k" * 16, "m" * 16]
    vb = np.stack([v.reshape(-1), full.reshape(-1)])
    c, o = D2.sample_cols(vb, toks, 5, 0, D2.STREAM_NOW, 32, nv=nv)
    for b, t in enumerate(toks):
        e = D2.type_balanced_cols(vb[b].reshape(16, nv), 32, token_rng(t, 5, 0, D2.STREAM_NOW))
        assert np.array_equal(c[b], e[0]) and np.array_equal(o[b], e[1])
    with pytest.raises(ValueError):
        D2.sample_cols(vb, toks, 5, 0, D2.STREAM_NOW, 32)                   # no nv for a non-96 width
    # lat4: short parents (S_8 < 3 m) have no valid lateral variant -> fewer than 32 valid -> padding (ok False)
    if arm == "lat4":
        vs = full.copy()
        vs[:10, 1:] = False                                                 # 6 long parents x 4 = 24 valid
        cols, ok = D2.type_balanced_cols(vs, 32, np.random.default_rng(0))
        assert ok.sum() == 24 and vs.reshape(-1)[cols].all()


# ----------------------------------------------------------------------------------------------- warm-up files
def _wu_dirs(root: Path, arm: str, n: int = 6, seed: int = 0, short_rows=(1,)):
    """structurally valid n-row warm-up sources for an arm (variant files made by online_variants on the anchors, so
    they hold the real validity pattern); build.json with the true provenance."""
    from tools.ck.data import common as CM
    L = lay(arm)
    nv, g_k = L.nv, L.g()
    pk, rl, vd = root / "pk", root / "rl", root / f"vd_{arm}"
    for d in (pk, rl / "shards", vd):
        d.mkdir(parents=True, exist_ok=True)
    toks = [f"t{i:015d}" for i in range(n)]
    sha = CM.sha16_array(np.array(toks, "U16"))
    g = np.random.default_rng(seed)
    anc = AS.load_anchors()
    if not (pk / "tokens.parquet").is_file():
        pd.DataFrame({"token": toks, "log": ["L"] * n, "row": np.arange(n)}).to_parquet(pk / "tokens.parquet")
        np.save(pk / "ok.npy", np.ones(n, bool))
        np.save(pk / "gt_traj.npy", anc[g.choice(256, n)].astype(np.float32))
        st = np.zeros((n, 8), np.float32)
        st[:, 4] = g.uniform(0, 12, n)
        np.save(pk / "status.npy", st)
        np.save(rl / "labels.npy", g.random((n, 256, 9)).astype(np.float32))
        np.save(rl / "ok.npy", np.ones((n, 256), bool))
        (pk / "meta.json").write_text(json.dumps({"token_sha16": sha}))
        (rl / "shards" / "_plan.json").write_text(json.dumps({"token_sha16": sha}))
    aidx = np.stack([np.random.default_rng(seed + i).choice(256, 16, replace=False) for i in range(n)]).astype(np.int16)
    par = torch.from_numpy(anc[aidx.astype(np.int64)].copy())
    for r in short_rows:
        par[r, :10] = par[r, :10] * 0.05                                    # short parents: laterals invalid
    v0 = torch.from_numpy(np.load(pk / "status.npy")[:, 4].astype(np.float64))
    V = C2.online_variants(par, v0, dict(vcfg(arm)))
    np.save(vd / "traj.npy", V["traj96"].numpy())
    np.save(vd / "labels.npy", g.random((n, g_k, 9)).astype(np.float32))
    np.save(vd / "ok.npy", np.ones((n, g_k), bool))
    np.savez(vd / "index.npz", valid=V["valid96"].numpy().reshape(n, 16, nv), anchor_idx=aidx,
             built=np.ones(n, bool), names=np.array(L.names))
    smp = AS.AnchorSampler(seed=0).config()
    build = {"names": list(L.names), "n_variants": nv, "n_anchors": 16, "speeds": list(L.speeds),
             "lats": list(L.lats), "combine": "separate", "s_on_frac": 0.2, "ext": "straight", "seed": 0,
             "sampler": smp, "anchors_sha16": AS.ANCHORS_SHA16, "n_tokens": n, "token_sha16": sha}
    (vd / "build.json").write_text(json.dumps(build))
    w = D2.warmup_cfg({"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)})
    return w, toks, V


@pytest.mark.parametrize("arm", ARMS)
def test_warmup_sources_per_arm(tmp_path, arm):
    L = lay(arm)
    w, toks, V = _wu_dirs(tmp_path, arm)
    src = D2._WUArrays(w, layout=L)
    chk = src.check()
    assert chk["n_rows"] == 6 and chk["var_build"]["names"] == list(L.names) and chk["provenance"]["sampler"] == "equal"
    if arm == "base":
        assert "SHAPES" not in src.__dict__ and D2._WUArrays(w).check()["n_rows"] == 6       # default = base
    else:
        # the base layout (default) refuses an arm's files structurally -- never an empty row
        with pytest.raises(D2.WUSourceError):
            D2.wu_row(D2._WUArrays(w), 0)
        with pytest.raises(D2.WUSourceError):
            D2._WUArrays(w).check()
    for other in ARMS:
        if other != arm and lay(other).nv == L.nv:                          # same width (v4 / spd4 / lat4)
            with pytest.raises(D2.WUSourceError, match="names"):
                D2._WUArrays(w, layout=lay(other)).check()
    row = D2.wu_row(src, 1)
    assert row["ck2_var_traj"].shape == (L.g(), 8, 3) and row["ck2_var_valid"].shape == (L.g(),)
    assert np.array_equal(row["ck2_var_valid"], V["valid96"][1].numpy()) and bool(row["ck2_wu_ok"])
    assert np.array_equal(row["ck2_var_traj"], V["traj96"][1].numpy())
    for k, v in D2.wu_row(src, -1).items():                                 # empty rows have the layout's width
        assert v.shape == D2.empty_wu_row(L.g())[k].shape and not v.any()
    # target builder from a config (layout from ck_e2e2.variants)
    b = D2.CK2E2ETargetBuilder({"use_teachers": False, "variants": vcfg(arm), "warmup": dict(w)})
    assert b.vnames == L.names and b.check()["n_rows"] == 6
    t = b.warmup_row(2)
    assert t["ck2_var_y"].shape == (L.g(), 5) and t["ck2_var_ok"].dtype == torch.bool
    if arm != "base":
        with pytest.raises(D2.WUSourceError):
            D2.CK2E2ETargetBuilder({"use_teachers": False, "warmup": dict(w)}).warmup_row(2)
    # prior: the arm's own files (base files would be refused)
    p = D2.ck2_warmup_prior({"variants": vcfg(arm), "warmup": dict(w)}, n_max=6)
    assert p.shape == (5,) and np.isfinite(p).all()


def _targets(builder, rows):
    outs = [builder.warmup_row(int(r)) for r in rows]
    return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}


@pytest.mark.parametrize("arm", ARMS)
def test_sampler_warmup_cands_fallback_per_arm(tmp_path, arm):
    L = lay(arm)
    nv = L.nv
    w, toks, _ = _wu_dirs(tmp_path, arm)
    b = D2.CK2E2ETargetBuilder({"use_teachers": False, "variants": vcfg(arm), "warmup": dict(w)})
    t = _targets(b, range(6))
    ws = D2.WarmupSampler.from_cfg({"variants": vcfg(arm), "warmup": dict(w)})
    assert ws.nv == nv and ws.g_k == L.g()
    S = ws.sample(toks, t, 2)
    assert S["var_cols"].shape == (6, 32) and (S["var_cols"] < L.g()).all() and (S["var_cols"] % nv != 0).all()
    assert np.array_equal(S["n_valid"], t["ck2_var_valid"].numpy().reshape(6, 16, nv)[:, :, 1:].sum((1, 2)))
    for bb, tok in enumerate(toks):
        e = D2.type_balanced_cols(t["ck2_var_valid"][bb].numpy().reshape(16, nv), 32, token_rng(tok, 2, 0, D2.STREAM_WU))
        assert np.array_equal(S["var_cols"][bb], e[0]) and np.array_equal(S["var_pad_ok"][bb], e[1])
    # the anchor draw does not depend on the variant layout
    Sb = D2.WarmupSampler().sample(toks, {k: (v if k != "ck2_var_valid" else torch.ones(6, 96, dtype=torch.bool))
                                          for k, v in t.items()}, 2)
    assert np.array_equal(S["anc_idx"], Sb["anc_idx"]) and np.array_equal(S["group"], Sb["group"])
    W = C2.warmup_cands(t, toks, 2, ws)
    vc = W["var_cols"].numpy()
    for bb in range(6):
        assert np.array_equal(W["cand"][bb, 32:].numpy(), t["ck2_var_traj"][bb].numpy()[vc[bb]])
        assert np.array_equal(W["vtype"][bb, 32:].numpy(), vc[bb] % nv)
        assert np.array_equal(W["parent_anchor"][bb, 32:].numpy(), t["ck2_var_anchor"][bb].numpy()[vc[bb] // nv])
        names = [L.names[c % nv] for c in vc[bb]]
        assert "id" not in names
    if arm != "base":
        with pytest.raises(ValueError, match="16 x 6"):
            C2.warmup_cands(t, toks, 2, D2.WarmupSampler())                 # base sampler vs the arm's targets
        with pytest.raises(ValueError):
            D2.WarmupSampler().sample(toks, t, 2)
    # fallback label set
    store = D2.LabelStore2(tmp_path / "no_io", n_rows=6, packed=w["packed"], layout=L)
    store.refresh(3)
    G = store.lookup(torch.arange(6), toks, epoch=4, seed=0)
    G = C2.fill_fallback(G, t, toks, 4, ws)
    Sl = ws.sample(toks, t, 4, stream=D2.STREAM_LAB)
    for bb in range(6):
        assert np.array_equal(G["G_vtype"][bb, 16:].numpy(), Sl["var_cols"][bb] % nv)
        assert np.array_equal(G["G_traj"][bb, 16:].numpy(), t["ck2_var_traj"][bb].numpy()[Sl["var_cols"][bb]])


def test_lat4_padding_short_parents(tmp_path):
    """lat4 has no speed variants: a token with >= 9 short parents has < 32 valid columns -> padded slots."""
    w, toks, V = _wu_dirs(tmp_path, "lat4", short_rows=(1, 2))
    vv = V["valid96"].numpy().reshape(6, 16, 5)
    n_val = vv[:, :, 1:].sum((1, 2))
    assert vv[1, :10, 1:].sum() == 0 and n_val[1] <= 24 and n_val[0] >= 32   # <= 6 long parents x 4 laterals
    b = D2.CK2E2ETargetBuilder({"use_teachers": False, "variants": vcfg("lat4"), "warmup": dict(w)})
    t = _targets(b, range(6))
    ws = D2.WarmupSampler(layout=lay("lat4"))
    S = ws.sample(toks, t, 0)
    assert S["var_pad_ok"][1].sum() == n_val[1] and S["var_pad_ok"][0].all() and S["n_valid"][1] == n_val[1]


# ----------------------------------------------------------------------------------------------- real label dirs
def _real(split, arm):
    root = Path(D2.REF_DATA_ROOT) / "ck"
    w = {"packed": str(root / "packed" / split), "raw_labels": str(root / "labels" / split / "raw256"),
         "var_dir": str(root / "ck2" / "labels" / split / arm_var_name(arm))}
    need = [Path(w["packed"]) / "tokens.parquet", Path(w["raw_labels"]) / "labels.npy"] + \
        [Path(w["var_dir"]) / f for f in ("build.json", "traj.npy", "labels.npy", "ok.npy", "index.npz")]
    return D2.warmup_cfg(w), all(p.is_file() for p in need)


@pytest.mark.parametrize("split,arm", [("navtrain_val", "v8"), ("navtrain_val", "v4"), ("navtrain_train", "v4"),
                                       ("navtrain_train", "v8")])
def test_real_arm_label_dirs_shared_columns_equal_base(split, arm):
    """the arm's warm-up label files through _WUArrays / wu_row / WarmupSampler / warmup_cands on ~300 real rows: every
    variant column whose name is a base name is bitwise the base file's column (traj, CK targets, ok, valid); the
    anchors, raw labels and GT are the same rows; online_variants(arm) == the arm's traj.npy / valid."""
    wa, have_a = _real(split, arm)
    wb, have_b = _real(split, "base")
    if not (have_a and have_b):
        pytest.skip(f"{split} {arm} / base warm-up label files not complete")
    La, Lb = lay(arm), BASE_LAYOUT
    sa, sb = D2._WUArrays(wa, layout=La), D2._WUArrays(wb)
    chk = sa.check()
    assert chk["var_build"]["names"] == list(La.names)
    with pytest.raises(D2.WUSourceError):
        D2._WUArrays(wa).check()
    C2.check_variant_files(wa["var_dir"], vcfg(arm))
    n = sa.n_rows()
    z = np.load(Path(wa["var_dir"]) / "index.npz")
    inval = np.flatnonzero(~z["valid"].all((1, 2)))
    rows = np.unique(np.concatenate([np.linspace(0, n - 1, 260).astype(np.int64), inval[:40]]))
    ra = [D2.wu_row(sa, int(r)) for r in rows]
    rb = [D2.wu_row(sb, int(r)) for r in rows]
    shared = [(La.names.index(nm), Lb.names.index(nm)) for nm in La.names if nm in Lb.names]
    assert len(shared) == {"v8": 6, "v4": 5}[arm]
    ca = np.array([k * La.nv + va for k in range(16) for va, _ in shared])
    cb = np.array([k * Lb.nv + vb for k in range(16) for _, vb in shared])
    for x, y in zip(ra, rb):
        for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_raw_ok", "ck2_var_anchor"):
            assert np.asarray(x[k]).tobytes() == np.asarray(y[k]).tobytes(), k
        for k in ("ck2_var_traj", "ck2_var_y", "ck2_var_ok", "ck2_var_valid"):
            assert x[k].shape[0] == La.g() and y[k].shape[0] == 96
            assert x[k][ca].tobytes() == y[k][cb].tobytes(), k
    # sampler / warm-up candidates of the arm on these rows
    rm = ED.RowMap(wa["packed"])
    toks = [rm.token_log(int(r))[0] for r in rows]
    t = {k: torch.stack([torch.from_numpy(np.array(x[k], copy=True)) for x in ra]) for k in ra[0]}
    ws = D2.WarmupSampler(layout=La)
    S = ws.sample(toks, t, 1)
    tbase = {k: torch.stack([torch.from_numpy(np.array(x[k], copy=True)) for x in rb]) for k in rb[0]}
    Sb = D2.WarmupSampler().sample(toks, tbase, 1)
    assert np.array_equal(S["anc_idx"], Sb["anc_idx"]) and np.array_equal(S["ok"], Sb["ok"])
    W = C2.warmup_cands(t, toks, 1, ws)
    ok_tok = S["ok"]
    vc = W["var_cols"].numpy()
    assert (vc[ok_tok] % La.nv != 0).all() and W["var_pad_ok"][torch.from_numpy(ok_tok)].all()   # >= 32 valid
    full = (t["ck2_var_valid"].numpy().reshape(len(rows), 16, La.nv)[:, :, 1:].sum((1,)) >= 32 // (La.nv - 1)).all(1)
    for bb in np.flatnonzero(ok_tok & full)[:50]:
        cnt = np.bincount(vc[bb] % La.nv, minlength=La.nv)
        assert cnt[0] == 0 and (cnt[1:] == 32 // (La.nv - 1)).all()         # type-balanced: v8 4 / v4 8 per type
    # online variants of the parents == the arm's label files (same make_variants path, float64 v0)
    st = np.asarray(np.load(Path(wa["packed"]) / "status.npy", mmap_mode="r")[rows[:40]])
    anc = AS.load_anchors()
    par = torch.from_numpy(anc[z["anchor_idx"][rows[:40]].astype(np.int64)])
    V = C2.online_variants(par, C2.v0_from_status(torch.from_numpy(st)), vcfg(arm))
    tr = np.asarray(np.load(Path(wa["var_dir"]) / "traj.npy", mmap_mode="r")[rows[:40]])
    assert V["traj96"].numpy().tobytes() == tr.tobytes()
    assert np.array_equal(V["valid96"].numpy(), z["valid"][rows[:40]].reshape(40, -1))


# ----------------------------------------------------------------------------------------------- rec2 / generations
def _rec_arrays(arm, n=3, seed=0):
    L = lay(arm)
    g = np.random.default_rng(seed)
    traj = g.normal(size=(n, L.g(), 8, 3)).astype(np.float32)
    valid = g.random((n, L.g())) > 0.1
    valid[:, ::L.nv] = True
    return traj, valid


@pytest.mark.parametrize("arm", ARMS)
def test_rec2_chunk_and_recorder_per_arm(tmp_path, arm):
    L = lay(arm)
    traj, valid = _rec_arrays(arm)
    p = tmp_path / "c.npz"
    D2.write_rec2_chunk(p, [1, 2, 3], traj, valid, np.zeros((3, 16)), np.ones((3, 16)), np.ones(3), 5, 4, 0,
                        layout=L)
    with np.load(p) as zz:
        keys = set(zz.files)
    if arm == "base":
        assert keys == OLD_REC2_KEYS                                      # base chunks: the old key set exactly
        q = tmp_path / "c_nolayout.npz"
        D2.write_rec2_chunk(q, [1, 2, 3], traj, valid, np.zeros((3, 16)), np.ones((3, 16)), np.ones(3), 5, 4, 0)
        assert q.read_bytes() == p.read_bytes()
    else:
        assert keys == OLD_REC2_KEYS | {"vnames"}
    a = D2.read_rec2_chunk(p)
    assert D2.rec2_vnames(a) == L.names and a["traj"].shape == (3, L.g(), 8, 3)
    assert np.array_equal(a["traj"], traj) and np.array_equal(a["valid"], valid)
    D2.read_rec2_chunk(p, layout=L)
    for other in ARMS:
        if other != arm:
            with pytest.raises(ValueError, match="variant layout"):
                D2.read_rec2_chunk(p, layout=lay(other))
    # a width that is not 16 x V of the declared layout is refused (also the old base check)
    with pytest.raises(ValueError):
        D2.write_rec2_chunk(tmp_path / "x.npz", [0], traj[:1, :32], valid[:1, :32], np.zeros((1, 16)),
                            np.zeros((1, 16)), np.zeros(1), 0, 4, 0, layout=L)
    rec = D2.CandRecorder2(tmp_path / "io", 0, 1, 4, chunk_tokens=2, attempt="abcd0001", layout=L)
    assert rec.add(torch.tensor([5, -1, 7]), torch.from_numpy(traj), torch.from_numpy(valid), torch.zeros(3, 16),
                   torch.zeros(3, 16), torch.ones(3), 9) == 2
    rec.close_epoch()
    c0 = D2.read_rec2_chunk(Path(ED.rec_rank_dir(tmp_path / "io", 4, 0)) / rec.chunks[0], layout=L)
    assert c0["row"].tolist() == [5, 7] and np.array_equal(c0["traj"], traj[[0, 2]])
    bad_w = 96 if arm != "base" else 80
    with pytest.raises(ValueError, match="CandRecorder2"):
        D2.CandRecorder2(tmp_path / "io2", 0, 1, 4, layout=L).add([0], np.zeros((1, bad_w, 8, 3), np.float32),
                                                                   np.ones((1, bad_w), bool), np.zeros((1, 16)),
                                                                   np.zeros((1, 16)), np.zeros(1), 0)


@pytest.mark.parametrize("arm", ARMS)
def test_generation_meta_and_layout_checks(tmp_path, arm):
    L = lay(arm)
    io = tmp_path / "io"
    gen = D2.open_generation2(io, 4, n_rows=8, mode="w+", layout=L)
    meta = json.loads((ED.gen_dir(io, 4) / "meta.json").read_text())
    assert meta["k"] == L.g() and meta["vnames"] == list(L.names) and gen["traj"].shape == (8, L.g(), 8, 3)
    if arm == "base":
        assert meta["format"] == "ck2_96" and meta["order"] == OLD_GEN2_ORDER
        assert list(meta) == ["epoch", "n_rows", "k", "format", "order", "vnames", "label_cols", "files",
                              "row_state", "created", "utc"]
        assert meta["files"] == {n: [list(s), np.dtype(t).name] for n, (s, t) in OLD_GEN2_FILES.items()}
        io_b = tmp_path / "io_default"
        D2.open_generation2(io_b, 4, n_rows=8, mode="w+")                   # default (no layout) = the same files
        for f in ("traj", "labels", "cand_ok", "valid", "row_state"):
            assert (ED.gen_dir(io_b, 4) / f"{f}.npy").read_bytes() == (ED.gen_dir(io, 4) / f"{f}.npy").read_bytes()
    else:
        assert meta["format"] == f"ck2_16x{L.nv}" != "ck2_96"
    traj, valid = _rec_arrays(arm, 2)
    lab = np.random.default_rng(1).random((2, L.g(), 9)).astype(np.float32)
    D2.write_generation_rows2(gen, [2, 5], traj, lab, np.ones((2, L.g()), bool), valid)
    with pytest.raises(AssertionError):
        D2.write_generation_rows2(gen, [1], traj[:, :L.g() - 16][:1], lab[:1, :L.g() - 16],
                                  np.ones((1, L.g() - 16), bool), valid[:1, :L.g() - 16])
    g2 = D2.open_generation2(io, 4, n_rows=8, mode="r")                     # no layout: the meta's own
    assert np.array_equal(g2["traj"][5], traj[1]) and g2["row_state"][[2, 5]].tolist() == [1, 1]
    D2.open_generation2(io, 4, n_rows=8, mode="r", layout=L.names)
    for other in ARMS:
        if other != arm:
            with pytest.raises(D2.GenLayoutError):
                D2.open_generation2(io, 4, n_rows=8, mode="r", layout=lay(other))
            with pytest.raises(D2.GenLayoutError):
                D2.open_generation2(io, 4, n_rows=8, mode="w+", layout=lay(other))   # reopen of an existing gen
    # inconsistent / CK1 metas are refused
    mp = ED.gen_dir(io, 4) / "meta.json"
    m = json.loads(mp.read_text())
    mp.write_text(json.dumps(dict(m, format="ck2_96" if arm != "base" else "ck2_16x6")))
    with pytest.raises(ValueError, match="CK1 generation"):
        D2.open_generation2(io, 4, n_rows=8, mode="r")


@pytest.mark.parametrize("arm", ARMS)
def test_label_store_per_arm(tmp_path, arm):
    L = lay(arm)
    nv = L.nv
    io = tmp_path / "io"
    w, toks, V = _wu_dirs(tmp_path, arm, n=8)
    gen = D2.open_generation2(io, 4, n_rows=8, mode="w+", layout=L)
    rows = np.array([1, 3, 6])
    tr = V["traj96"].numpy()[rows]
    va = V["valid96"].numpy()[rows]
    lab = np.random.default_rng(3).random((3, L.g(), 9)).astype(np.float32)
    cok = np.ones((3, L.g()), bool)
    cok[0, 1] = False
    D2.write_generation_rows2(gen, rows, tr, lab, cok, va)
    st = D2.LabelStore2(io, n_rows=8, packed=w["packed"], layout=L)
    r = st.refresh(4)
    assert r["gens"] == [4] and r["n_prev"] == 3
    G = st.lookup(torch.tensor([1, 3, 6, 0, -1]), None, epoch=5, seed=0, n_var=32)
    assert G["has"].tolist() == [True, True, True, False, False]
    for b, rr in enumerate(rows):
        col = G["G_col"][b].numpy().astype(np.int64)
        assert np.array_equal(col[:16], np.arange(16) * nv)                          # identities k * V
        e = D2.type_balanced_cols((va[b] & cok[b]).reshape(16, nv), 32, token_rng(toks[rr], 5, 0, D2.STREAM_LAB))
        assert np.array_equal(col[16:], e[0])
        assert np.array_equal(G["G_vtype"][b].numpy(), (col % nv).astype(np.int8))
        assert np.array_equal(G["G_traj"][b].numpy(), tr[b][col])
    # a store of another layout refuses the generation (same width included)
    for other in ARMS:
        if other != arm:
            with pytest.raises(D2.GenLayoutError):
                D2.LabelStore2(io, n_rows=8, packed=w["packed"], layout=lay(other)).refresh(4)


# ----------------------------------------------------------------------------------------------- labeler2
def _tokens_parquet(tmp: Path, n=12):
    p = tmp / "tokens.parquet"
    pd.DataFrame(dict(token=[f"tok{i:04d}" for i in range(n)], log=["__fake__"] * n, row=np.arange(n))).to_parquet(
        p, index=False)
    return str(p)


@pytest.mark.parametrize("arm", ("v8", "v4", "lat4"))
def test_labeler2_fake_scorer_per_arm(tmp_path, arm):
    from tools.ck.e2e import labeler as L1
    from tools.ck.e2e2 import labeler2 as LB2
    L = lay(arm)
    io = tmp_path / "ck_e2e2"
    tok = _tokens_parquet(tmp_path)
    traj, valid = _rec_arrays(arm, 3, seed=2)
    p = ED.rec_chunk_path(io, 5, 0, "aaaa0000", 0)
    D2.write_rec2_chunk(p, [1, 4, 6], traj, valid, np.zeros((3, 16)), np.ones((3, 16)), np.ones(3), 0, 5, 0, layout=L)
    # a base chunk of the same epoch: another arm's record -> BAD chunk, never mixed into the generation
    tb, vb = _rec_arrays("base", 1, seed=3)
    pb = ED.rec_chunk_path(io, 5, 0, "aaaa0000", 1)
    D2.write_rec2_chunk(pb, [8], tb, vb, np.zeros((1, 16)), np.ones((1, 16)), np.ones(1), 0, 5, 0)
    lb = LB2.Labeler2(io, 1, workers=2, poll=0.2, once=True, scorer="fake", n_rows=12, status_every=0, tokens=tok,
                      log=lambda s: None)
    assert lb.run() == 0
    meta = json.loads((ED.gen_dir(io, 5) / "meta.json").read_text())
    assert meta["vnames"] == list(L.names) and meta["k"] == L.g() and meta["format"] == D2.gen2_format(L)
    g = D2.open_generation2(io, 5, 12, "r", layout=L)
    assert g["row_state"][[1, 4, 6, 8]].tolist() == [1, 1, 1, 0]
    for i, r in enumerate([1, 4, 6]):                                   # ALL 16 V recorded trajectories scored
        assert np.array_equal(g["traj"][r], traj[i]) and np.array_equal(g["valid"][r], valid[i])
        assert np.array_equal(np.asarray(g["labels"][r]).view(np.uint8), L1.fake_scores(traj[i]).view(np.uint8))
        assert g["cand_ok"][r].all()
    errs = [json.loads(x) for x in (ED.gen_dir(io, 5) / "errors.jsonl").read_text().splitlines()]
    assert len(errs) == 1 and "bad chunk" in errs[0]["error"] and "variant layout" in errs[0]["error"]
    st = D2.LabelStore2(io, n_rows=12, packed=str(tmp_path), layout=L)
    assert st.refresh(5)["n_prev"] == 3
    with pytest.raises(D2.GenLayoutError):
        D2.LabelStore2(io, n_rows=12, packed=str(tmp_path)).refresh(5)


# ----------------------------------------------------------------------------------------------- online2
class _FakeTeachers:
    """TeacherPair2 stand-in: every real candidate has a KD target (kd_ok = cand_ok)."""
    calib_info: dict = {}

    def to(self, device):
        return self

    def run(self, bev_det, ok_det, bev_map, ok_map, cand, status, cand_ok=None, extra=None):
        B, K = cand.shape[:2]
        ok = torch.ones(B, K, dtype=torch.bool) if cand_ok is None else cand_ok.bool().clone()
        z = torch.zeros(B, K, 6)
        p = torch.full((B, K, 5), 0.5)
        return {"kd_score_prob": p, "kd_e_lat": z, "kd_ok": ok, "e_det": z, "e_map": z, "p_det": p, "p_map": p,
                "z_det": torch.zeros(B, K, 5), "z_map": torch.zeros(B, K, 5), "ok_det": ok, "ok_map": ok}


def _packed_tiny(tmp: Path, n=16):
    pk = tmp / "packed"
    pk.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"token": [f"p{i:015d}" for i in range(n)], "log": ["L"] * n, "row": np.arange(n)}).to_parquet(
        pk / "tokens.parquet")
    return str(pk)


def _ck(tmp_path, arm, **over):
    d = {"enabled": True, "io_dir": str(tmp_path / f"io_{arm}"), "teacher_det_run": "/nonexistent/T",
         "teacher_map_run": "/nonexistent/M", "score_prior": "none", "grad_share_every": 0, "strict_rows": 0.0,
         "kd_calib": {"enabled": False}, "teacher_amp": False, "rec_chunk_tokens": 1, "lat_kd": {"start_mb": 0},
         "variants": vcfg(arm), "warmup": {"packed": _packed_tiny(tmp_path)}}
    d.update(over)
    c = O.CKE2E2Config.from_any(d)
    ck = O.CKE2E2(c, max_epochs=30)
    ck._teachers = _FakeTeachers()
    return c, ck


def _syn_targets(arm, B, seed=0, short=()):
    """synthetic warm-up targets of an arm's width (variant files = online variants of random anchors)."""
    from navsim.agents.para_ssr.refiner.adapters import BEV_H, BEV_W
    L = lay(arm)
    g = torch.Generator().manual_seed(seed)
    anc = torch.from_numpy(AS.load_anchors())
    va = torch.stack([torch.randperm(256, generator=g)[:16] for _ in range(B)])
    par = anc[va].clone()
    for b in short:
        par[b, :10] *= 0.05
    V = C2.online_variants(par, torch.full((B,), 5.0, dtype=torch.float64), vcfg(arm))
    t = {"ck2_wu_ok": torch.ones(B, dtype=torch.bool), "ck2_gt": anc[torch.randint(0, 256, (B,), generator=g)],
         "ck2_raw_y": torch.rand(B, 256, 5, generator=g), "ck2_raw_ok": torch.ones(B, 256, dtype=torch.bool),
         "ck2_var_traj": V["traj96"], "ck2_var_y": torch.rand(B, L.g(), 5, generator=g),
         "ck2_var_ok": torch.ones(B, L.g(), dtype=torch.bool), "ck2_var_valid": V["valid96"],
         "ck2_var_anchor": va.long()}
    t["trajectory"] = t["ck2_gt"].clone()
    for i in (0, 1):
        t[f"kd_bev_{i}"] = torch.zeros(B, 256, BEV_H, BEV_W, dtype=torch.float16)
        t[f"kd_ok_{i}"] = torch.ones(B, dtype=torch.bool)
    return t


def _pred(B, seed=0, short_top=0):
    """synthetic v2 predictions; short_top: the top-ranked short_top candidates are made short (S_8 < 3 m)."""
    from navsim.agents.para_ssr.refiner.adapters import BEV_H, BEV_W
    g = torch.Generator().manual_seed(seed)
    anc = torch.from_numpy(AS.load_anchors()).float()
    off = 0.1 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    if short_top:
        top = final.topk(short_top, dim=-1).indices
        for b in range(B):
            off[b, top[b], :, :2] = -0.97 * anc[top[b], :, :2]
    return {"bev_embed": torch.randn(B, BEV_H * BEV_W, 256, generator=g).requires_grad_(True),
            "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anc,
            "im_rewards": torch.softmax(torch.randn(B, 256, generator=g), -1),
            "sim_rewards": torch.sigmoid(torch.randn(B, 5, 256, generator=g))}


@pytest.fixture(scope="module")
def student():
    c = O.CKE2E2Config.from_any({"enabled": True, "score_prior": "none"})
    return O.build_student_ck2(c, apply_prior=False)


@pytest.mark.parametrize("arm", ARMS)
def test_cke2e2_loss_record_onpolicy_infer_per_arm(tmp_path, student, arm):
    L = lay(arm)
    c, ck = _ck(tmp_path, arm)
    assert ck.layout == L
    B = 2
    t = _syn_targets(arm, B)
    t["ck_row"] = torch.tensor([3, 7])
    f = {"status_feature": torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0.5, 0, 0]])}
    # warm-up
    ck.epoch, ck.epoch_frac = 0, 0.1
    loss, logs = ck.loss(student, f, t, _pred(B, 1))
    assert torch.isfinite(loss) and float(logs["ck2/n_cand"]) == 64.0 and float(logs["ck2/kd_ok_frac"]) == 1.0
    vv = t["ck2_var_valid"].reshape(B, 16, L.nv)[:, :, 1:].float().mean()
    assert float(logs["ck2/var_valid_frac"]) == pytest.approx(float(vv), abs=1e-6)
    # warmup_record: the recorder writes the arm's 16 V pool (vnames for a non-base layout)
    ck.epoch, ck.epoch_frac = 4, 4.1
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 4, 1, layout=ck.layout)
    loss, logs = ck.loss(student, f, t, _pred(B, 2))
    assert float(logs["ck2/rec_tokens"]) == 2.0
    a = D2.read_rec2_chunk(Path(ED.rec_rank_dir(c.io_dir, 4, 0)) / ck.recorder.chunks[0], layout=L)
    assert a["traj"].shape == (1, L.g(), 8, 3) and D2.rec2_vnames(a) == L.names
    ck.recorder = None
    # on-policy: generation 4 of the arm's layout for row 3; row 7 -> warm-up fallback
    gen = D2.open_generation2(c.io_dir, 4, n_rows=16, mode="w+", layout=L)
    D2.write_generation_rows2(gen, [3], a["traj"], np.random.default_rng(0).random((1, L.g(), 9)).astype(np.float32),
                              np.ones((1, L.g()), bool), a["valid"])
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], layout=ck.layout)
    ck.epoch, ck.epoch_frac = 5, 5.5
    loss, logs = ck.loss(student, f, t, _pred(B, 3))
    assert torch.isfinite(loss) and float(logs["ck2/n_cand"]) == 96.0
    assert float(logs["ck2/G_src_prev"]) == 0.5 and float(logs["ck2/G_src_fallback"]) == 0.5
    assert float(logs["ck2/kd_ok_frac"]) == 1.0
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    # inference over the 16 V pool, with the selection rules resolved through the layout
    student.eval()
    try:
        pr = _pred(B, 4)
        res = ck.infer(student, f, pr)
        assert res["ck2_cand96"].shape == (B, L.g(), 8, 3) and res["ck2_score_logit"].shape == (B, L.g(), 5)
        assert torch.equal(res["ck2_cand96"][:, ::L.nv], ck.infer(student, f, pr)["ck2_cand96"][:, ::L.nv])
        for sel in ({"beta": 1.0, "set": "all", "lat_mode": "on"},
                    {"beta": 0.5, "set": "lat", "lat_mode": "on_except_latvar", "rule": "wote"},
                    {"beta": 0.7, "set": "decel", "lat_mode": "off"}):
            ck.cfg.infer_select = sel
            r2 = ck.infer(student, f, pr)
            i = r2["ck2_sel_idx"]
            assert i.shape == (B,) and int(i.max()) < L.g() and torch.isfinite(r2["ck2_traj"]).all()
            vt = (i % L.nv).tolist()
            assert all(v in L.sets[sel["set"]] for v in vt), (sel, vt)
            if sel["lat_mode"] == "off":
                exp = res["ck2_cand96"][torch.arange(B), i]
                assert torch.equal(r2["ck2_traj"], exp)
    finally:
        ck.cfg.infer_select = None
        student.train()


def test_kd_ok_frac_excludes_padded_slots_lat4(tmp_path, student, monkeypatch):
    """lat4 on-policy with >= 9 short top-16 parents: fewer than 32 valid lateral columns, padded slots are not KD
    candidates (kd_ok_frac over the real candidates = 1 with every teacher target present); base: no padding, the
    old expression."""
    B = 2
    f = {"status_feature": torch.tensor([[0, 0, 0, 0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0.1, 0, 0, 0]])}
    out = {}
    for arm in ("lat4", "base"):
        c, ck = _ck(tmp_path / arm, arm)
        t = _syn_targets(arm, B)
        t["ck_row"] = torch.tensor([-1, -1])
        ck._labels = D2.LabelStore2(tmp_path / arm / "no_io", n_rows=16, packed=c.warmup["packed"], layout=ck.layout)
        ck.epoch, ck.epoch_frac = 5, 5.0
        seen = {}
        orig = ck.current_batch

        def spy(*a, **k):
            seen["S"] = orig(*a, **k)
            return seen["S"]
        ck.current_batch = spy
        _, logs = ck.loss(student, f, t, _pred(B, 5, short_top=12))
        S = seen["S"]
        n_pad = int(S["pad_slot"].sum())
        out[arm] = (n_pad, float(logs["ck2/kd_ok_frac"]))
        if arm == "lat4":
            assert n_pad > 0 and float(logs["ck2/kd_ok_frac"]) == 1.0
            assert int(S["cand_ok"].sum()) == S["cand_ok"].numel() - n_pad
        else:
            assert n_pad == 0 and float(logs["ck2/kd_ok_frac"]) == 1.0
    # _check_inputs: padded slots do not count against strict_rows (base: n_pad 0 = the old counts)
    monkeypatch.setattr(O, "TOKENS_CHECK", 2)
    c, ck = _ck(tmp_path / "chk", "lat4", strict_rows=0.999)
    kd_ok = torch.ones(2, 48, dtype=torch.bool)
    kd_ok[:, 40:] = False                                             # 8 padded slots per token
    ck._check_inputs(torch.ones(2, dtype=torch.bool), kd_ok, n_pad=16)
    assert ck._inputs["kd_all"] == 80 and ck._inputs_checked
    c, ck = _ck(tmp_path / "chk2", "lat4", strict_rows=0.999)
    with pytest.raises(RuntimeError, match="kd_ok_frac"):
        ck._check_inputs(torch.ones(2, dtype=torch.bool), kd_ok)          # counted as missing without n_pad


# ----------------------------------------------------------------------------------------------- review fixes
def test_valid2d_base_shapes_keep_the_old_reshape():
    """nv None: every 96-element valid (any shape) and every [..., 6] valid draws as v2_final's reshape(-1, 6) (a
    [1, 96] / [2, 48] valid used to be read with 95 / 47 'types', identity columns included); a [16, V] valid infers V;
    any other shape of an arm needs nv."""
    g = np.random.default_rng(7)
    for i in range(40):
        v = g.random(96) > g.choice([0.05, 0.4, 0.9])
        vb = g.random((2, 16, 6)) > g.choice([0.05, 0.4, 0.9])
        for vv in (v.reshape(1, 96), v.reshape(2, 48), v.reshape(4, 24), v.reshape(8, 12), v.reshape(2, 8, 6), vb):
            for n in (0, 1, 32, 33, 90):
                a = D2.type_balanced_cols(vv, n, np.random.default_rng(i * 31 + n))
                b = _old_type_balanced_cols(vv, n, np.random.default_rng(i * 31 + n))
                assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)), (i, n, vv.shape)
                a = D2.uniform_cols(vv, n, np.random.default_rng(i + n))
                b = _old_uniform_cols(vv, n, np.random.default_rng(i + n))
                assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)), (i, n, vv.shape)
    for arm in ("v8", "v4", "spd4", "lat4"):
        L = lay(arm)
        cols, ok = D2.type_balanced_cols(np.ones((16, L.nv), bool), 32, np.random.default_rng(0))
        assert ok.all() and (cols % L.nv != 0).all() and cols.max() < L.g()
        for shape in ((L.g(),), (1, L.g()), (2, 8, L.nv)):
            with pytest.raises(ValueError, match="pass nv"):
                D2.type_balanced_cols(np.ones(shape, bool), 32, np.random.default_rng(0))
            with pytest.raises(ValueError, match="pass nv"):
                D2.uniform_cols(np.ones(shape, bool), 32, np.random.default_rng(0))


def _nan_top_pred(B, seed, n_nan):
    """_pred with the n_nan top-ranked candidates of every token non-finite (NaN student offsets)."""
    pr = _pred(B, seed)
    top = pr["plan_final_rewards"].topk(n_nan, dim=-1).indices
    for b in range(B):
        pr["trajectory_offset"][b, top[b]] = float("nan")
    return pr


def test_kd_ok_frac_base_counts_unchanged_with_nonfinite_parents(tmp_path, student):
    """base table: non-finite student parents leave fewer than n_var valid variant columns (padded slots exist), yet
    the KD counts stay the v2_final ones (kd_ok_frac = kd_ok.mean(), _check_inputs gets n_pad 0, so strict_rows still
    counts these broken tokens); a non-base layout (lat4) leaves the padded slots out."""
    B = 2
    f = {"status_feature": torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0], [0, 0, 0, 0, 4.0, 0.2, 0, 0]])}
    for arm in ("base", "lat4"):
        c, ck = _ck(tmp_path / arm, arm)
        t = _syn_targets(arm, B)
        t["ck_row"] = torch.tensor([-1, -1])
        ck._labels = D2.LabelStore2(tmp_path / arm / "no_io", n_rows=16, packed=c.warmup["packed"], layout=ck.layout)
        ck.epoch, ck.epoch_frac = 5, 5.0
        seen = {}
        orig_cb, orig_ci = ck.current_batch, ck._check_inputs

        def spy_cb(*a, **k):
            seen["S"] = orig_cb(*a, **k)
            return seen["S"]

        def spy_ci(ref, kd_ok, n_pad=0):
            seen["kd_ok"], seen["n_pad"] = kd_ok.clone(), n_pad
            return orig_ci(ref, kd_ok, n_pad)
        ck.current_batch, ck._check_inputs = spy_cb, spy_ci
        _, logs = ck.loss(student, f, t, _nan_top_pred(B, 5, 12))
        pad = int(seen["S"]["pad_slot"].sum())
        kd = seen["kd_ok"]
        got = float(logs["ck2/kd_ok_frac"])
        assert pad > 0 and 0 < int(kd.sum()) < kd.numel(), (arm, pad, int(kd.sum()))
        if arm == "base":
            assert seen["n_pad"] == 0 and got == float(kd.float().mean())
            assert got != pytest.approx(float(kd.sum()) / (kd.numel() - pad))
        else:
            assert seen["n_pad"] == pad and got == pytest.approx(float(kd.sum()) / (kd.numel() - pad))


@pytest.mark.parametrize("arm", ARMS)
def test_launcher_warmup_check_uses_the_layout(tmp_path, arm):
    """launch_util2.warmup_check (experiment.sh check / smoke / train via build_overrides2) checks the warm-up files
    against the layout of ck_e2e2.variants: the arm's own files pass, another arm's files (same width included) are a
    LaunchError; base: the record of the old call (_WUArrays without a layout)."""
    from tools.ck.e2e2 import launch_util2 as L2
    L = lay(arm)
    w, _, _ = _wu_dirs(tmp_path, arm)
    srcs = {k: w[k] for k in ("packed", "raw_labels", "var_dir")}
    c = O.CKE2E2Config.from_any({"variants": vcfg(arm), "warmup": srcs})
    rec = L2.warmup_check(c)
    assert rec["n_rows"] == 6 and rec["var_build"]["names"] == list(L.names) and rec["provenance"]["sampler"] == "equal"
    if arm == "base":
        old = D2._WUArrays(D2.warmup_cfg(c.to_dict()), c.ep_target).check()
        assert json.dumps(rec, sort_keys=True, default=str) == json.dumps(old, sort_keys=True, default=str)
    for other in ARMS:
        if other != arm:
            co = O.CKE2E2Config.from_any({"variants": vcfg(other), "warmup": srcs})
            with pytest.raises(L2.LaunchError, match="ck_e2e2.warmup sources"):
                L2.warmup_check(co)


@pytest.mark.parametrize("arm", ("v8", "v4"))
def test_launcher_warmup_check_real_arm_dirs(arm):
    from tools.ck.e2e2 import launch_util2 as L2
    d = _train_dir(arm)
    if not (d / "build.json").is_file() or not (Path(D2.WARMUP_DEFAULT["var_dir"]) / "build.json").is_file():
        pytest.skip("navtrain_train arm / base label dirs absent")
    rec = L2.warmup_check(O.CKE2E2Config.from_any({"variants": vcfg(arm), "warmup": {"var_dir": str(d)}}))
    assert rec["n_rows"] == D2.N_ROWS and rec["provenance"]["sampler"] == "equal"
    other = "v4" if arm == "v8" else "v8"
    for cfg in ({"variants": vcfg(other), "warmup": {"var_dir": str(d)}}, {"warmup": {"var_dir": str(d)}},
                {"variants": vcfg(arm)}):
        with pytest.raises(L2.LaunchError, match="ck_e2e2.warmup sources"):
            L2.warmup_check(O.CKE2E2Config.from_any(cfg))
