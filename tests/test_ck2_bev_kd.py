"""CK2 simple BEV feature KD (navsim/agents/para_ssr/ck/bev_kd.py) tests, CPU.

Synthetic: layout round trip (= ck.online.bev_sgrid), GTLoader key mapping, adapter inits (zero / identity / default,
no global RNG), loss = hand formula (mse / cosine), ok mask, cell weight, two-teacher mean, gradient paths, f16 teacher
input under autocast, gradient-norm helper, share <-> weight algebra and ShareController.
Real data (skipped when the caches are absent): v2dump r34 student bev + BEVFusion t0 / ReSMap teacher BEVs + the CK
teacher-run norm files on a few navtrain_train tokens: z-score statistics, loss levels at init, adapter learning,
lateral-flip content check (the S grids agree).

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_bev_kd.py
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.ck import bev_kd as BK

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
NORM_T = CK / "train/ckT_p1/norm.npz"
NORM_M = CK / "train/ckM_p1/norm_map.npz"
V2DUMP = CK / "v2dump/navtrain_train/bev"
TOKENS = CK / "packed/navtrain_train/tokens.parquet"


def _norm(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.3, 1.0, 256).astype(np.float32)


def _batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    bev = torch.randn(B, BK.N_CELLS, BK.BEV_C, generator=g)
    tb = torch.relu(torch.randn(B, 256, 50, 100, generator=g)).half()
    return bev, tb


def _manual(bev, tb, mean, std, W, b, distance, ok, ln=True, clip=None):
    x = bev.double()
    if ln:
        x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
    p = x @ W.double().T + b.double()
    z = (tb.double().flatten(2).transpose(1, 2) - torch.as_tensor(mean).double()) / torch.as_tensor(std).double()
    if clip is not None:
        z = z.clamp(-clip, clip)
    if distance == "mse":
        d = (p - z).pow(2).mean(-1)
    else:
        d = 1 - (p * z).sum(-1) / (p.norm(dim=-1).clamp(min=1e-6) * z.norm(dim=-1).clamp(min=1e-6))
    return d.mean(1)[ok].mean()


# ----------------------------------------------------------------------------------------------- layout / keys
def test_layout_round_trip_matches_online():
    from navsim.agents.para_ssr.ck.online import bev_sgrid

    bev, _ = _batch(2)
    sg = BK.tokens_to_sgrid(bev)
    assert torch.equal(sg, bev_sgrid(bev))
    assert torch.equal(BK.sgrid_to_tokens(sg), bev)
    for r, c in ((0, 0), (12, 37), (49, 99), (25, 0)):
        assert torch.equal(sg[:, :, r, c], bev[:, r * 100 + c])
    with pytest.raises(ValueError):
        BK.sgrid_to_tokens(torch.zeros(1, 256, 100, 50))
    with pytest.raises(ValueError):
        BK.tokens_to_sgrid(torch.zeros(1, 4999, 256))


def test_teacher_keys_follow_teacher_run_order():
    assert BK.teacher_keys(["T", "M"]) == {"det": ("kd_bev_0", "kd_ok_0"), "map": ("kd_bev_1", "kd_ok_1")}
    assert BK.teacher_keys(["M", "T"]) == {"map": ("kd_bev_0", "kd_ok_0"), "det": ("kd_bev_1", "kd_ok_1")}
    assert BK.teacher_keys(["T"]) == {"det": ("kd_bev_0", "kd_ok_0")}
    with pytest.raises(ValueError):
        BK.teacher_keys(["T", "T"])
    with pytest.raises(ValueError):
        BK.teacher_keys(["S"])
    tg = {"kd_bev_0": torch.zeros(1), "kd_ok_0": torch.ones(1)}
    bev, ok = BK.teacher_inputs(tg, BK.teacher_keys(["T"]), ["det"])
    assert bev["det"] is tg["kd_bev_0"] and ok["det"] is tg["kd_ok_0"]
    with pytest.raises(KeyError):
        BK.teacher_inputs(tg, BK.teacher_keys(["T"]), ["map"])


# ----------------------------------------------------------------------------------------------- module basics
def test_param_count_buffers_and_config():
    kd = BK.BEVFeatureKD(["det", "map"], {"det": _norm(0), "map": _norm(1)})
    assert kd.param_count() == 2 * (256 * 256 + 256)
    names = {n for n, _ in kd.named_parameters()}
    assert names == {f"adapters.{t}.proj.{w}" for t in ("det", "map") for w in ("weight", "bias")}
    sd = kd.state_dict()
    assert "zscore.det.mean" in sd and "zscore.map.std" in sd            # z-score saved, never trained
    assert not any(k.startswith("w_init") for k in sd)                   # diagnostic reference not saved
    assert kd.config()["params"] == 131_584 and kd.config()["init"] == "zero"
    with pytest.raises(ValueError):
        BK.BEVFeatureKD([], {})
    with pytest.raises(ValueError):
        BK.BEVFeatureKD(["det"], {"map": _norm(0)})
    with pytest.raises(ValueError):
        BK.BEVFeatureKD(["det"], {"det": _norm(0)}, distance="l1")
    with pytest.raises(ValueError):
        BK.BEVFeatureKD(["det", "det"], {"det": _norm(0)})


def test_inits_and_no_global_rng():
    torch.manual_seed(123)
    a = torch.rand(3)
    torch.manual_seed(123)
    BK.BEVFeatureKD(["det", "map"], {"det": _norm(0), "map": _norm(1)}, init="default")
    assert torch.equal(torch.rand(3), a)
    bev, _ = _batch(2)
    z = BK.BEVKDAdapter("zero")
    assert torch.count_nonzero(z(bev)) == 0
    i = BK.BEVKDAdapter("identity")
    assert torch.allclose(i(bev), F.layer_norm(bev, (256,)), atol=1e-6)
    assert torch.allclose(BK.BEVKDAdapter("identity", ln=False)(bev), bev, atol=1e-6)
    d1, d2 = BK.BEVKDAdapter("default", seed=5), BK.BEVKDAdapter("default", seed=5)
    assert torch.equal(d1.proj.weight, d2.proj.weight)
    assert not torch.equal(d1.proj.weight, BK.BEVKDAdapter("default", seed=6).proj.weight)


@pytest.mark.parametrize("distance", ["mse", "cosine"])
@pytest.mark.parametrize("clip", [None, 1.5])
def test_loss_matches_hand_formula(distance, clip):
    mean, std = _norm(3)
    kd = BK.BEVFeatureKD(["det"], {"det": (mean, std)}, distance=distance, init="default", target_clip=clip)
    bev, tb = _batch(3)
    ok = torch.tensor([True, False, True])
    out = kd(bev, {"det": tb}, {"det": ok})
    W, b = kd.adapters["det"].proj.weight.detach(), kd.adapters["det"].proj.bias.detach()
    ref = _manual(bev, tb.float(), mean, std, W, b, distance, ok, clip=clip)
    assert abs(float(out["loss"]) - float(ref)) < 1e-5 * max(1.0, abs(float(ref)))
    assert abs(float(out["raw/det"]) - float(ref)) < 1e-5 * max(1.0, abs(float(ref)))
    assert float(out["ok_frac/det"]) == pytest.approx(2 / 3)


def test_ok_mask_and_all_missing():
    kd = BK.BEVFeatureKD(["det"], {"det": _norm(0)}, init="identity")
    bev, tb = _batch(3)
    ok = torch.tensor([True, False, True])
    tb2 = tb.clone()
    tb2[1] = tb2[1] * 7 + 3
    l1 = kd(bev, {"det": tb}, {"det": ok})["loss"]
    l2 = kd(bev, {"det": tb2}, {"det": ok})["loss"]
    assert float(l1) == float(l2)
    b = bev.clone().requires_grad_(True)
    out = kd(b, {"det": tb}, {"det": ok})
    out["loss"].backward()
    assert torch.count_nonzero(b.grad[1]) == 0 and torch.count_nonzero(b.grad[0]) > 0
    kd.zero_grad()
    b = bev.clone().requires_grad_(True)
    out = kd(b, {"det": tb}, {"det": torch.zeros(3, dtype=torch.bool)})
    assert float(out["loss"]) == 0.0 and float(out["raw/det"]) == 0.0
    out["loss"].backward()
    for p in kd.parameters():
        assert p.grad is not None and torch.count_nonzero(p.grad) == 0
    assert b.grad is None


def test_cell_weight():
    kd = BK.BEVFeatureKD(["det"], {"det": _norm(0)}, init="default")
    bev, tb = _batch(2)
    ok = torch.ones(2, dtype=torch.bool)
    l0 = kd(bev, {"det": tb}, {"det": ok})["loss"]
    l1 = kd(bev, {"det": tb}, {"det": ok}, cell_weight=torch.full((2, 50, 100), 3.0))["loss"]
    assert float(l0) == pytest.approx(float(l1), rel=1e-6)
    w = torch.zeros(2, 50, 100)
    w[:, :10] = 1.0                                       # first 10 rows (0-6.4 m) only
    lw = kd(bev, {"det": tb}, {"det": ok}, cell_weight=w)["loss"]
    m, s = kd.zscore["det"].mean, kd.zscore["det"].std
    with torch.no_grad():
        p = kd.adapters["det"](bev)
        z = (BK.sgrid_to_tokens(tb.float()) - m) / s
        ref = (p - z).pow(2).mean(-1)[:, :1000].mean()
    assert float(lw) == pytest.approx(float(ref), rel=1e-5)
    with pytest.raises(ValueError):
        kd(bev, {"det": tb}, {"det": ok}, cell_weight=-w)


def test_two_teachers_mean_and_gradient_paths():
    nd, nm = _norm(0), _norm(1)
    both = BK.BEVFeatureKD(["det", "map"], {"det": nd, "map": nm}, init="identity")
    only_d = BK.BEVFeatureKD(["det"], {"det": nd}, init="identity")
    only_m = BK.BEVFeatureKD(["map"], {"map": nm}, init="identity")
    bev, tb = _batch(2)
    _, tm = _batch(2, seed=1)
    ok = torch.ones(2, dtype=torch.bool)
    lb = both(bev, {"det": tb, "map": tm}, {"det": ok, "map": ok})["loss"]
    ld = only_d(bev, {"det": tb}, {"det": ok})["loss"]
    lm = only_m(bev, {"map": tm}, {"map": ok})["loss"]
    assert float(lb) == pytest.approx(0.5 * (float(ld) + float(lm)), rel=1e-6)
    b = bev.clone().requires_grad_(True)
    t = tb.float().requires_grad_(True)
    both(b, {"det": t, "map": tm}, {"det": ok, "map": ok})["loss"].backward()
    assert t.grad is None                                 # teacher side: no gradient
    assert b.grad is not None and torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0
    assert all(p.grad is not None for p in both.parameters())


def test_per_teacher_inputs_and_losses():
    """{t: input} (the e2e arm's per-teacher GradScale points) == one shared tensor; loss/<t> carries each teacher's
    own graph (its adapter + its input only); loss == mean_t loss/<t>; masked tokens counted in n_ok/<t>."""
    nd, nm = _norm(0), _norm(1)
    kd = BK.BEVFeatureKD(["det", "map"], {"det": nd, "map": nm}, init="identity")
    bev, tb = _batch(3)
    _, tm = _batch(3, seed=1)
    ok_d = torch.tensor([True, True, True])
    ok_m = torch.tensor([True, False, True])
    shared = kd(bev, {"det": tb, "map": tm}, {"det": ok_d, "map": ok_m})
    xd, xm = bev.clone().requires_grad_(True), bev.clone().requires_grad_(True)
    per = kd({"det": xd, "map": xm}, {"det": tb, "map": tm}, {"det": ok_d, "map": ok_m})
    for k in ("loss", "loss/det", "loss/map", "raw/det", "raw/map", "fve/map"):
        assert torch.equal(shared[k].detach(), per[k].detach()), k
    assert torch.equal(per["loss"], 0.5 * (per["loss/det"] + per["loss/map"])) or \
        float(per["loss"]) == pytest.approx(0.5 * float(per["loss/det"] + per["loss/map"]), rel=1e-6)
    assert int(per["n_ok/det"]) == 3 and int(per["n_ok/map"]) == 2 and float(per["ok_frac/map"]) == pytest.approx(2 / 3)
    assert float(per["loss/map"]) == pytest.approx(float(_manual(bev, tm, *nm, torch.eye(256), torch.zeros(256), "mse",
                                                                 ok_m)), rel=1e-5)
    g = torch.autograd.grad(per["loss/det"], [xd, xm] + list(kd.adapters["det"].parameters()) +
                            list(kd.adapters["map"].parameters()), allow_unused=True)
    assert g[0] is not None and g[1] is None and all(x is not None for x in g[2:4]) and g[4] is None and g[5] is None
    gm = torch.autograd.grad(per["loss/map"], xm)[0]
    assert float(gm[1].abs().max()) == 0.0 and float(gm[0].abs().max()) > 0           # masked token: no gradient
    none = kd({"det": xd, "map": xm}, {"det": tb, "map": tm}, {"det": ok_d, "map": torch.zeros(3, dtype=torch.bool)})
    assert float(none["loss/map"]) == 0.0 and none["loss/map"].requires_grad and int(none["n_ok/map"]) == 0
    gz = torch.autograd.grad(none["loss/map"], list(kd.adapters["map"].parameters()))
    assert all(float(x.abs().max()) == 0.0 for x in gz)                                # in the graph, zero gradient
    with pytest.raises(ValueError, match="batch size"):
        kd({"det": xd, "map": xm[:2]}, {"det": tb, "map": tm}, {"det": ok_d, "map": ok_m})


def test_zero_init_has_no_bev_gradient_at_step0():
    kd = BK.BEVFeatureKD(["det"], {"det": _norm(0)}, init="zero")
    bev, tb = _batch(2)
    b = bev.clone().requires_grad_(True)
    out = kd(b, {"det": tb}, {"det": torch.ones(2, dtype=torch.bool)})
    g = BK.bev_grad_norms({"kd": out["loss"]}, b)["kd"]
    assert g == 0.0
    z = (BK.sgrid_to_tokens(tb.float()) - kd.zscore["det"].mean) / kd.zscore["det"].std
    assert float(out["loss"]) == pytest.approx(float(z.pow(2).mean()), rel=1e-5)
    out["loss"].backward()
    assert kd.adapters["det"].proj.weight.grad.abs().sum() > 0       # the adapter learns first


def test_half_teacher_under_autocast_and_shape_errors():
    kd = BK.BEVFeatureKD(["det"], {"det": _norm(0)}, init="identity")
    bev, tb = _batch(2)
    ok = torch.ones(2, dtype=torch.bool)
    ref = kd(bev, {"det": tb.float()}, {"det": ok})["loss"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = kd(bev, {"det": tb}, {"det": ok})
    assert out["loss"].dtype == torch.float32 and float(out["loss"]) == pytest.approx(float(ref), rel=1e-6)
    with pytest.raises(ValueError):
        kd(bev[:, :100], {"det": tb}, {"det": ok})
    with pytest.raises(ValueError):
        kd(bev, {"det": tb[:, :, :, :50]}, {"det": ok})
    with pytest.raises(ValueError):
        kd(bev, {"det": tb}, {"det": torch.ones(3, dtype=torch.bool)})


# ----------------------------------------------------------------------------------------------- share helpers
def test_bev_grad_norms_matches_manual_and_keeps_graph():
    lin = torch.nn.Linear(256, 1)
    bev = torch.randn(2, 5000, 256, requires_grad=True)
    l1 = lin(bev).pow(2).mean()
    l2 = bev.sum() * 0.5
    g = BK.bev_grad_norms({"a": l1, "b": l2, "none": None}, bev)
    ref = torch.autograd.grad(l1, bev, retain_graph=True)[0].norm()
    assert g["a"] == pytest.approx(float(ref), rel=1e-6)
    assert g["b"] == pytest.approx(0.5 * math.sqrt(bev.numel()), rel=1e-6)
    assert g["none"] == 0.0
    assert bev.grad is None and lin.weight.grad is None
    (l1 + l2).backward()                                  # graph still usable
    assert bev.grad is not None


def test_share_weight_algebra():
    for rho in (0.05, 0.1, 0.5):
        w = BK.weight_for_share(rho, 2e-3, 0.06)
        assert BK.share_of(w, 2e-3, 0.06) == pytest.approx(rho, rel=1e-12)
    assert BK.weight_for_share(0.1, 1e-3, 0.09) == pytest.approx(0.1 / 0.9 * 90, rel=1e-12)
    with pytest.raises(ValueError):
        BK.weight_for_share(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        BK.weight_for_share(0.1, 0.0, 1.0)


def test_share_controller():
    c = BK.ShareController(0.1, w_init=5.0, m=0.5, w_min=0.1, w_max=100.0)
    assert c.weight == 5.0
    assert c.update(0.0, 0.06) == 5.0 and c.n_skip == 1           # zero-init adapter: skipped
    assert c.update(float("nan"), 0.06) == 5.0 and c.n_skip == 2
    w = c.update(1e-3, 0.06)                                       # first valid measurement adopted
    assert w == pytest.approx(0.1 / 0.9 * 60, rel=1e-12)
    assert BK.share_of(w, 1e-3, 0.06) == pytest.approx(0.1, rel=1e-12)
    w2 = c.update(1e-3, 0.24)                                      # ratio x4 -> log-EMA, m = 0.5 -> x2
    assert w2 == pytest.approx(0.1 / 0.9 * 120, rel=1e-9)
    for _ in range(200):
        c.update(2e-3, 0.06)
    assert BK.share_of(c.weight, 2e-3, 0.06) == pytest.approx(0.1, rel=1e-6)
    sd = c.state_dict()
    c2 = BK.ShareController(0.1, w_init=5.0, m=0.5, w_min=0.1, w_max=100.0)
    c2.load_state_dict(sd)
    assert c2.weight == c.weight and c2.n == c.n
    hi = BK.ShareController(0.5, w_init=1.0, w_max=10.0)
    assert hi.update(1e-6, 1.0) == 10.0                            # clamp


# ----------------------------------------------------------------------------------------------- real data
def _real_ok():
    return NORM_T.is_file() and NORM_M.is_file() and V2DUMP.is_dir() and TOKENS.is_file()


real = pytest.mark.skipif(not _real_ok(), reason="CK caches / v2dump not on this machine")


def _real_tokens(n_logs, seed=0):
    import pandas as pd

    t = pd.read_parquet(TOKENS)
    rng = np.random.default_rng(seed)
    logs = rng.choice(np.array(sorted(t.log.unique())), n_logs, replace=False)
    return [t[t.log == lg].token.iloc[0] for lg in logs]


def _load(tok):
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache

    s = np.load(V2DUMP / tok[:2] / f"{tok}.npy").astype(np.float32)          # S grid [256, 50, 100]
    bev = torch.from_numpy(s.reshape(256, -1).T.copy())                     # bev_embed layout [5000, 256]
    d = torch.from_numpy(np.asarray(TeacherCache.for_subset("navtrain").load_bev(tok, s_grid=True)))
    m = torch.from_numpy(np.asarray(ResmapCache.for_subset("navtrain").load_bev(tok, s_grid=True)))
    return bev, d, m


@pytest.fixture(scope="module")
def real4():
    if not _real_ok():
        pytest.skip("no real data")
    toks = _real_tokens(4, seed=11)
    items = [_load(t) for t in toks]
    bev = torch.stack([i[0] for i in items])
    det = torch.stack([i[1] for i in items])
    mp = torch.stack([i[2] for i in items])
    norms = {"det": BK.load_teacher_norm(NORM_T), "map": BK.load_teacher_norm(NORM_M)}
    return bev, det, mp, norms


@real
def test_real_inputs_layout_and_zscore(real4):
    bev, det, mp, norms = real4
    assert det.dtype == torch.float16 and det.shape == (4, 256, 50, 100) and mp.shape == (4, 256, 50, 100)
    cell_std = bev.std(-1, unbiased=False)
    assert 0.9 < float(cell_std.mean()) < 1.1                      # bev_embed already per-cell normalised
    assert float(bev.mean(-1).abs().max()) < 0.1
    kd = BK.BEVFeatureKD(["det", "map"], norms)
    for t, x in (("det", det), ("map", mp)):
        z = kd.zscore[t](BK.sgrid_to_tokens(x))
        assert torch.isfinite(z).all()
        assert abs(float(z.mean())) < 0.5 and 0.5 < float(z.pow(2).mean()) < 2.0


@real
def test_real_ck2_teacher_run_norms(real4):
    """The z-score files the e2e BEV-KD arm reads (bev_kd.NORM_FILE in the ck_e2e2 teacher runs: det ->
    <teacher_det_run>/norm.npz, map -> <teacher_map_run>/norm_map.npz; yaml defaults ck2T10dep / ck2M10dep, and the
    official-EP ck2T10 / ck2M10) load and z-score real BEVFusion t0 / ReSMap BEVs to ~zero mean, ~unit energy."""
    from navsim.agents.para_ssr.ck.e2e_data2 import TEACHER_DET_RUN, TEACHER_MAP_RUN
    assert BK.NORM_FILE == {"det": "norm.npz", "map": "norm_map.npz"}
    _, det, mp, _ = real4
    runs = [("det", Path(TEACHER_DET_RUN), det), ("map", Path(TEACHER_MAP_RUN), mp),
            ("det", CK / "ck2/train/ck2T10", det), ("map", CK / "ck2/train/ck2M10", mp)]
    seen = 0
    for t, run, x in runs:
        if not (run / BK.NORM_FILE[t]).is_file():
            continue
        seen += 1
        mean, std = BK.norm_from_run(run, t)
        z = BK.TeacherZ(mean, std)(BK.sgrid_to_tokens(x))
        assert torch.isfinite(z).all() and abs(float(z.mean())) < 0.5 and 0.5 < float(z.pow(2).mean()) < 2.0, (run, t)
    if not seen:
        pytest.skip("no CK2 teacher run norm file on this machine")


@real
def test_real_loss_levels_at_init(real4):
    bev, det, mp, norms = real4
    ok = torch.ones(4, dtype=torch.bool)
    for t, x in (("det", det), ("map", mp)):
        zero = BK.BEVFeatureKD([t], {t: norms[t]}, init="zero")(bev, {t: x}, {t: ok})
        ident = BK.BEVFeatureKD([t], {t: norms[t]}, init="identity")(bev, {t: x}, {t: ok})
        # zero init: mean z^2 ~ 1;  identity: ~ 1 + 1 (unrelated channel pairing, measured 2.0 on 60 held-out logs)
        assert 0.5 < float(zero["loss"]) < 1.6, float(zero["loss"])
        assert 1.5 < float(ident["loss"]) < 2.6, float(ident["loss"])
        assert float(ident[f"fve/{t}"]) < 0.0                          # identity pairing explains nothing
        cos = BK.BEVFeatureKD([t], {t: norms[t]}, init="identity", distance="cosine")(bev, {t: x}, {t: ok})
        assert 0.8 < float(cos["loss"]) < 1.2                          # ~ orthogonal


@real
def test_real_adapter_learns_and_bev_gradient_flows(real4):
    bev, det, _, norms = real4
    ok = torch.ones(4, dtype=torch.bool)
    torch.manual_seed(0)
    kd = BK.BEVFeatureKD(["det"], {"det": norms["det"]}, init="zero")
    opt = torch.optim.Adam(kd.parameters(), lr=3e-3)
    first = None
    for _ in range(40):
        out = kd(bev, {"det": det}, {"det": ok})
        if first is None:
            first = float(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    last = kd(bev, {"det": det}, {"det": ok})
    assert float(last["loss"]) < first - 0.05, (first, float(last["loss"]))
    assert float(last["fve/det"]) > 0.05 and float(last["w_dev/det"]) > 0
    b = bev.clone().requires_grad_(True)
    g = BK.bev_grad_norms({"kd": kd(b, {"det": det}, {"det": ok})["loss"]}, b)["kd"]
    assert math.isfinite(g) and g > 0                                   # BEV gradient appears once W != 0


@real
def test_real_lateral_flip_breaks_content_fit():
    """1x1 ridge from LN(student) to z(teacher) after removing each map's per-cell TRAIN mean (so only scene content
    can be fitted): the S grids agree -> the unflipped teacher fits clearly better than the laterally flipped one
    (full run, 400 logs: det 0.078 vs 0.009, map 0.120 vs 0.042; ck2/bev_kd/analysis/probe_content.json)."""
    if not _real_ok():
        pytest.skip("no real data")
    toks = _real_tokens(48, seed=5)
    items = [_load(t) for t in toks]
    norms = {"det": BK.load_teacher_norm(NORM_T), "map": BK.load_teacher_norm(NORM_M)}
    S = torch.stack([F.layer_norm(i[0], (256,)) for i in items])                          # [N, 5000, 256]
    rng = np.random.default_rng(0)
    cells = torch.as_tensor(rng.choice(5000, 1200, replace=False))
    n_tr = 36
    for k, j in (("det", 1), ("map", 2)):
        mean, std = (torch.as_tensor(a).view(1, 1, 256) for a in norms[k])
        Z = (torch.stack([BK.sgrid_to_tokens(i[j][None].float())[0] for i in items]) - mean) / std
        Zf = (torch.stack([BK.sgrid_to_tokens(torch.flip(i[j], dims=[-1])[None].float())[0] for i in items]) - mean) / std
        r2 = {}
        for name, T in (("same", Z), ("flip", Zf)):
            Sc = S - S[:n_tr].mean(0, keepdim=True)
            Tc = T - T[:n_tr].mean(0, keepdim=True)
            X = Sc[:, cells].double()
            Y = Tc[:, cells].double()
            Xtr, Ytr = X[:n_tr].reshape(-1, 256), Y[:n_tr].reshape(-1, 256)
            A = Xtr.T @ Xtr
            B = torch.linalg.solve(A + 1e-2 * torch.trace(A) / 256 * torch.eye(256, dtype=torch.float64), Xtr.T @ Ytr)
            Xte, Yte = X[n_tr:].reshape(-1, 256), Y[n_tr:].reshape(-1, 256)
            r2[name] = float(1 - ((Xte @ B - Yte) ** 2).sum() / (Yte ** 2).sum())
        assert r2["same"] > r2["flip"] + 0.02, (k, r2)
