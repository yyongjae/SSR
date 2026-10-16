"""refiner v4 parent dedupe (docs/refiner_v4/V4_3_SPEC.md M3): CKNet.score_set runs trunk.stations once per UNIQUE
parent of a token and gathers its rows back to the candidates.  Checked against the repeated per-candidate parent pass
(score_set(..., dedupe=False), the v4_2 ops): max abs diff 0 on score_logit / w_lat / every 'haz' tensor / ctx for the
inference 96 pool (16 parents) and a training-like set (KD 16 + 32 variants, label set 16 + 32 variants: 32 / 24
unique parents, so padding is exercised), hazard and evidence readouts, eval and train mode; gradients equal up to the
summation order; the explicit parent_index API and the per-parent context re-use (correct_topk / corrector_logp) give
the same bits; the FLOPs of the parent pass drop by K / U.  CPU fp32, synthetic inputs, ~1 min.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_dedupe.py
"""
import pytest
import torch

from navsim.agents.para_ssr.ck.cands2 import online_variants
from navsim.agents.para_ssr.ck.model import CKNet
from navsim.agents.para_ssr.ck.v4 import V4Cfg
from navsim.agents.para_ssr.refiner.e2e import ego_inputs

T = 2
HAZ = V4Cfg(readout="hazard", pref_head=True)
EVI = V4Cfg(readout="evidence", pref_head=True)
OUT_KEYS = ("score_logit", "w_lat", "z_lon", "gate_logit", "haz", "ctx")


def _draft(v, kap):
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    s = v * t
    if abs(kap) < 1e-9:
        return torch.stack([s, 0 * s, 0 * s], -1)
    return torch.stack([torch.sin(kap * s) / kap, (1 - torch.cos(kap * s)) / kap, kap * s], -1)


def _top(v0, dv, kap):
    return torch.stack([torch.stack([_draft(v0 + dv * k, kap * (k - 8) + 1e-4) for k in range(16)])
                        for _ in range(T)])                                          # [T, 16, 8, 3] f64


def _sampled(t96, t16, g):
    """KD / label-set layout: the 16 identities (own parents) + 32 sampled variants (parent = their identity)."""
    k = torch.randint(0, 16, (T, 32), generator=g)
    v = torch.randint(1, 6, (T, 32), generator=g)
    var = torch.gather(t96, 1, (k * 6 + v)[..., None, None].expand(T, 32, 8, 3))
    vp = torch.gather(t16, 1, k[..., None, None].expand(T, 32, 8, 3))
    return torch.cat([t16, var], 1), torch.cat([t16, vp], 1)


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    v0 = torch.full((T,), 6.0, dtype=torch.float64)
    top = _top(6.0, 0.3, 0.01)
    lab = _top(5.0, 0.4, -0.008)
    lab[1, :8] = top[1, :8]                     # token 1: half of the label identities = KD identities (U 24 vs 32)
    traj96 = online_variants(top, v0, None)["traj96"].float()
    lab96 = online_variants(lab, v0, None)["traj96"].float()
    top, lab = top.float(), lab.float()
    g = torch.Generator().manual_seed(1)
    kc, kp = _sampled(traj96, top, g)
    lc, lp = _sampled(lab96, lab, g)
    status = torch.zeros(T, 8)
    status[:, 1] = 1
    status[:, 4] = 6.0
    return {"traj96": traj96, "par96": top.repeat_interleave(6, 1), "top": top, "status": status,
            "train_c": torch.cat([kc, lc], 1), "train_p": torch.cat([kp, lp], 1),
            "bev": torch.randn(T, 256, 50, 100)}


def _maxdiff(a, b, path=""):
    """max abs diff over (nested dicts of) tensors; AssertionError unless every entry is the same (NaN == NaN)."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        return max([_maxdiff(a[k], b[k], f"{path}/{k}") for k in a] + [0.0])
    assert a.shape == b.shape and a.dtype == b.dtype, path
    if a.dtype == torch.bool or not a.is_floating_point():
        assert torch.equal(a, b), path
        return 0.0
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    d = float((a.double() - b.double()).abs().nan_to_num(0.0, 0.0, 0.0).max()) if a.numel() else 0.0
    assert bool(same.all()), f"{path}: max abs diff {d}"
    return d


def _scene(net, data):
    ego = ego_inputs(data["status"])
    feat, mem = net.trunk.scene(data["bev"], T)
    return feat, mem, ego, net.scene_pref(mem, ego)


# ----------------------------------------------------------------------------------------------- unique rows
def test_unique_parents_bitwise():
    a = torch.randn(8, 3)
    b = torch.randn(8, 3)
    n = a.clone()
    n[3, 1] = float("nan")
    z = torch.zeros(8, 3)
    zn = z.clone()
    zn[0, 0] = -0.0                                                      # == +0.0 numerically, different bits
    par = torch.stack([torch.stack([a, b, a, n, n, z, zn, b]),
                       torch.stack([b, b, b, b, b, b, b, b])])          # [2, 8, 8, 3]
    pu, pidx, nu = CKNet.unique_parents(par)
    assert nu.tolist() == [5, 1] and tuple(pu.shape) == (2, 5, 8, 3)
    assert pidx.tolist() == [[0, 1, 0, 2, 2, 3, 4, 1], [0] * 8]           # first-occurrence order
    back = torch.gather(pu, 1, pidx[..., None, None].expand(2, 8, 8, 3))
    assert torch.equal(back.view(torch.int32), par.view(torch.int32))   # bit for bit (NaN row included)
    assert torch.equal(pu[1, 1:].view(torch.int32), b[None].expand(4, 8, 3).view(torch.int32))   # pad = row 0


# ----------------------------------------------------------------------------------------------- bit identity
@pytest.mark.parametrize("readout", ["hazard", "evidence"])
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_dedupe_is_bit_identical_pool96(data, readout, mode):
    """inference 96 pool: 16 parents (c // 6) -> one parent pass over 16 rows per token."""
    net = CKNet("S", seed=3, v4=HAZ if readout == "hazard" else EVI)
    getattr(net, mode)()
    with torch.no_grad():
        feat, mem, ego, pref = _scene(net, data)
        rep = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_ctx=True, dedupe=False)
        ded = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_ctx=True,
                            return_pctx=True)
        assert tuple(ded["pctx"]["parents"].shape) == (T, 16, 8, 3) and ded["pctx"]["x"].shape[0] == T * 16
        assert torch.equal(ded["parent_index"], (torch.arange(96) // 6)[None].expand(T, 96))
        assert max(_maxdiff(rep[k], ded[k], k) for k in OUT_KEYS) == 0.0
        # explicit API: unique parents [T, 16] + parent_index (no search) -> the same bits
        exp = net.score_set(feat, mem, ego, data["traj96"], data["top"], pref=pref, return_ctx=True,
                            parent_index=ded["parent_index"])
        assert max(_maxdiff(rep[k], exp[k], k) for k in OUT_KEYS) == 0.0
        # the plain attribute switches the default; return_pctx without dedupe = the repeated rows
        net.dedupe_parents = False
        off = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_pctx=True)
        assert off["pctx"]["x"].shape[0] == T * 96 and torch.equal(off["score_logit"], rep["score_logit"])


def test_dedupe_training_like_set_forward_and_grad(data):
    """KD 16 + 32 sampled variants and a label set 16 + 32 (other identities; token 1 shares 8 of them): U 32 / 24
    (padded to 32 for token 1).  Forward bit for bit in train mode with grad; gradients equal up to the order of
    the sums (the K candidates' gradients are summed before the parent pass)."""
    cand, par = data["train_c"], data["train_p"]
    pu, pidx, nu = CKNet.unique_parents(par)
    assert nu.tolist() == [32, 24] and pu.shape[1] == 32
    net = CKNet("S", seed=3, v4=HAZ).train()
    res = []
    for dd in (False, True):
        net.zero_grad(set_to_none=True)
        feat, mem, ego, pref = _scene(net, data)
        o = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, dedupe=dd)
        (o["score_logit"].pow(2).mean() + o["w_lat"].pow(2).mean() + torch.sigmoid(o["haz"]["a_nc"]).mean()
         + torch.sigmoid(o["haz"]["a_dac"]).mean() + pref.mean()).backward()
        res.append((o, {n: q.grad.clone() for n, q in net.named_parameters() if q.grad is not None}))
    (oa, ga), (ob, gb) = res
    assert max(_maxdiff(oa[k], ob[k], k) for k in OUT_KEYS) == 0.0
    assert set(ga) == set(gb) and any(n.startswith("trunk.enc.") for n in ga)
    for n in ga:
        scale = float(ga[n].abs().max())
        assert float((ga[n] - gb[n]).abs().max()) <= 1e-4 * scale + 1e-10, n


# ----------------------------------------------------------------------------------------------- context re-use
def test_pctx_reuse_correct_topk_and_logp(data):
    """correct_topk with pool_out from score_set(..., return_pctx=True) re-uses the pool's parent pass (no stations
    on the sources' parents); corrector_logp with parent_index + pctx likewise: bit for bit the re-run."""
    net = CKNet("S", seed=0, v4=V4Cfg(readout="hazard", pref_head=True, corrector=True, corr_k=4)).eval()
    with torch.no_grad():
        feat, mem, ego, pref = _scene(net, data)
        cand, par = data["traj96"], data["par96"]
        net.dedupe_parents = False                                       # the v4_2 path end to end
        po0 = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True)
        ce0 = net.correct_topk(feat, mem, ego, cand, par, po0, pref=pref, sample=True,
                               generator=torch.Generator().manual_seed(0))
        net.dedupe_parents = True
        po1 = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, return_pctx=True)
        ce1 = net.correct_topk(feat, mem, ego, cand, par, po1, pref=pref, sample=True,
                               generator=torch.Generator().manual_seed(0))
    for k in ("src_idx", "u", "mu", "log_sig", "logp", "edit_traj", "edit_score_logit", "edit_w_lat", "edit_haz",
              "ctx"):
        assert _maxdiff(ce0[k], ce1[k], k) == 0.0
    assert float((ce1["edit_traj"] - torch.gather(cand, 1, ce1["src_idx"][..., None, None].expand(T, 4, 8, 3)))
                 .abs().max()) > 1e-3                                    # the sampled edits really moved
    src = torch.gather(cand, 1, ce1["src_idx"][..., None, None].expand(T, 4, 8, 3))
    src_par = torch.gather(par, 1, ce1["src_idx"][..., None, None].expand(T, 4, 8, 3))
    net.dedupe_parents = False
    lp0 = net.corrector_logp(feat, mem, ego, src, src_par, ce1["u"], pref=pref)
    net.dedupe_parents = True
    lp1 = net.corrector_logp(feat, mem, ego, src, None, ce1["u"], pref=pref,
                             parent_index=torch.gather(po1["parent_index"], 1, ce1["src_idx"]), pctx=po1["pctx"])
    assert torch.equal(lp0, lp1) and torch.allclose(lp1, ce1["logp"], atol=1e-4)


def test_mean_readout_and_argument_checks(data):
    """mean readout: every v4 argument ignored (CK2 ops).  Misuse of the explicit / re-use API raises."""
    n0 = CKNet("S", seed=3)
    feat, mem, ego, _ = _scene(n0, data)
    with torch.no_grad():
        a = n0.score_set(feat, mem, ego, data["traj96"])
        b = n0.score_set(feat, mem, ego, data["traj96"], data["par96"], parent_index=torch.zeros(T, 96).long(),
                         return_pctx=True)
    assert torch.equal(a["score_logit"], b["score_logit"]) and "pctx" not in b
    net = CKNet("S", seed=3, v4=HAZ).eval()
    c = data["traj96"][:, :12]
    with torch.no_grad():
        feat, mem, ego, pref = _scene(net, data)
        po = net.score_set(feat, mem, ego, c, data["par96"][:, :12], return_pctx=True)
        assert po["pctx"]["parents"].shape[1] == 2
        with pytest.raises(ValueError, match="parent_index"):
            net.score_set(feat, mem, ego, c, pctx=po["pctx"])
        with pytest.raises(ValueError, match="parent_index"):
            net.score_set(feat, mem, ego, c, data["top"], parent_index=torch.full((T, 12), 16).long())
        with pytest.raises(ValueError, match="pctx"):
            net.score_set(feat, mem, ego, c, data["top"][:, :2] + 1.0, parent_index=po["parent_index"],
                          pctx=po["pctx"])
        with pytest.raises(ValueError, match="parents"):
            net.score_set(feat, mem, ego, c, data["par96"][:, :11])
        # re-use with the same parents passed explicitly = the plain call
        r = net.score_set(feat, mem, ego, c, po["pctx"]["parents"], parent_index=po["parent_index"],
                          pctx=po["pctx"])
        assert torch.equal(r["score_logit"], po["score_logit"])


# ----------------------------------------------------------------------------------------------- cost
def test_flops_drop_with_unique_parents(data):
    """FlopCounterMode (conv / matmul; the CPU attention kernel is not counted): the 96 pool's score_set drops by
    > 5x (16 of 96 parent rows; the per-candidate readout stays)."""
    from torch.utils.flop_counter import FlopCounterMode
    net = CKNet("S", seed=3, v4=HAZ).eval()
    with torch.no_grad():
        feat, mem, ego, pref = _scene(net, data)
        fl = []
        for dd in (False, True):
            fc = FlopCounterMode(display=False)
            with fc:
                net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, dedupe=dd)
            fl.append(fc.get_total_flops())
    assert fl[0] / fl[1] > 5.0, fl
