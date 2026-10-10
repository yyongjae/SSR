"""CK core losses (navsim/agents/para_ssr/ck/losses.py), CPU only.

  score_bce     ok / non-finite masking, value = manual BCE, 0 without ok candidates (graph kept)
  score_kd_bce  0 when student == teacher (entropy-corrected), same gradient as the plain soft BCE, > 0 otherwise
  ctrl_kd_l1    0 when equal; lon 0.25 / lat 1.0 weights; ok mask; full [.., 8] decode outputs accepted
  surrogate     correction_loss on real GT tokens (skipped without GT on disk): finite, reaches the lon / lat heads,
                token-subset restriction == surrogate on the subset only, a non-finite term drops the micro-batch
  lead_bce      has_lead / censored masking
"""
from __future__ import annotations


import pytest
import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.ck import constants as C
from navsim.agents.para_ssr.ck import losses as L
from navsim.agents.para_ssr.ck import model as M
from navsim.agents.para_ssr.refiner import e2e as E

GT_TOKENS = ("1aa44d46e4ab5bc7", "153c6b07f09d53d1")


def _cand(T=2, K=4, v=6.0, seed=0, noise=0.2):
    g = torch.Generator().manual_seed(seed)
    s = v * torch.arange(1, 9, dtype=torch.float32) * 0.5
    base = torch.stack([s, torch.zeros(8), torch.zeros(8)], -1)
    scale = torch.linspace(0.0, 1.0, 8)[:, None] * torch.tensor([noise, noise, 0.02])
    return (base[None, None].repeat(T, K, 1, 1) + torch.randn(T, K, 8, 3, generator=g) * scale).contiguous()


def _status(T=2, v=6.0):
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = v
    return sf


# ----------------------------------------------------------------------------------------------- score BCE
def test_score_bce_masks_and_matches_manual():
    g = torch.Generator().manual_seed(0)
    T, K = 3, 4
    logit = torch.randn(T, K, 5, generator=g, requires_grad=True)
    y = torch.rand(T, K, 5, generator=g).round()
    y[0, 0, 0] = 0.5                                                   # soft NC label
    ok = torch.ones(T, K, dtype=torch.bool)
    ok[1, 2] = False
    y[2, 3, 1] = float("nan")                                          # unscored value -> not ok
    loss, per = L.score_bce(logit, y, ok)
    m = ok.clone()
    m[2, 3] = False
    ref = F.binary_cross_entropy_with_logits(logit[m], y[m], reduction="none").mean(0)
    torch.testing.assert_close(loss, ref.mean())
    assert per == pytest.approx({k: float(ref[i].detach()) for i, k in enumerate(C.CK_KEYS)}, rel=1e-6)
    loss.backward()
    assert torch.equal(logit.grad[1, 2], torch.zeros(5)) and torch.isfinite(logit.grad).all()
    z, _ = L.score_bce(logit, y, torch.zeros(T, K, dtype=torch.bool))
    assert float(z.detach()) == 0.0 and z.requires_grad


def test_score_kd_zero_when_student_equals_teacher_and_plain_bce_gradient():
    g = torch.Generator().manual_seed(1)
    T, K = 2, 16
    lt = torch.randn(T, K, 5, generator=g) * 3
    p_t = torch.sigmoid(lt)
    ok = torch.ones(T, K, dtype=torch.bool)
    s = lt.clone().requires_grad_(True)
    kd = L.score_kd_bce(s, p_t, ok)
    assert abs(float(kd.detach())) < 1e-6
    # hard teacher probabilities (0 / 1) are fine too
    hard = (p_t > 0.5).float()
    assert float(L.score_kd_bce(torch.where(hard > 0, 30.0, -30.0), hard, ok)) < 1e-6
    # gradient = plain soft-BCE gradient (the entropy term is a constant)
    s2 = (lt + torch.randn(T, K, 5, generator=g)).requires_grad_(True)
    s3 = s2.detach().clone().requires_grad_(True)
    L.score_kd_bce(s2, p_t, ok).backward()
    F.binary_cross_entropy_with_logits(s3, p_t).backward()
    torch.testing.assert_close(s2.grad, s3.grad)
    assert float(L.score_kd_bce(s2.detach(), p_t, ok)) > 1e-3
    assert float(L.score_kd_bce(s2.detach(), p_t, ok, entropy_correct=False)) > float(L.score_kd_bce(s2.detach(), p_t, ok))


# ----------------------------------------------------------------------------------------------- control KD
def test_ctrl_kd_l1_weights_mask_and_zero():
    T, K = 2, 3
    cs, es = torch.zeros(T, K, 6), torch.zeros(T, K, 6)
    ok = torch.ones(T, K, dtype=torch.bool)
    l0, p0 = L.ctrl_kd_l1(cs, es, cs.clone(), es.clone(), ok)
    assert float(l0) == 0.0 and p0 == {"lon": 0.0, "lat": 0.0}
    l_lon, _ = L.ctrl_kd_l1(cs, es, cs - 1.0, es, ok)
    l_lat, _ = L.ctrl_kd_l1(cs, es, cs, es + 1.0, ok)
    assert float(l_lon) == pytest.approx(C.KD_CTRL_W["lon"]) and float(l_lat) == pytest.approx(C.KD_CTRL_W["lat"])
    ct = torch.zeros(T, K, 6)
    ct[0, 0] = -4.0
    ct[1, 1] = float("nan")
    ok2 = ok.clone()
    ok2[0, 0] = False
    l_m, p_m = L.ctrl_kd_l1(cs, es, ct, es, ok2)                                  # masked + non-finite -> 0
    assert float(l_m) == 0.0
    # full decode outputs [.., 8] are sliced to [2:]
    c8 = torch.cat([torch.zeros(T, K, 2), cs - 2.0], -1)
    l8, p8 = L.ctrl_kd_l1(torch.cat([torch.zeros(T, K, 2), cs], -1), torch.cat([torch.zeros(T, K, 2), es], -1), c8,
                          torch.cat([torch.zeros(T, K, 2), es], -1), ok)
    assert p8["lon"] == pytest.approx(2.0) and float(l8) == pytest.approx(0.5)


def test_lead_bce_masks():
    logit = torch.tensor([2.0, -1.0, 0.5, 3.0])
    has = torch.tensor([1.0, 1.0, 0.0, 1.0])
    d1 = torch.tensor([1.0, 0.0, 1.0, float("nan")])
    ref = F.binary_cross_entropy_with_logits(logit[:2], d1[:2])
    torch.testing.assert_close(L.lead_bce(logit, has, d1), ref)
    assert float(L.lead_bce(logit, torch.zeros(4), d1)) == 0.0


# ----------------------------------------------------------------------------------------------- surrogate
def _ref(tokens):
    ld = E.GTLoader(C.DATA_ROOT)
    items = [ld.load(t) for t in tokens]
    if not all(bool(it["ref_gt_ok"]) for it in items):
        pytest.skip("surrogate GT for the test tokens not available")
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def _student_out(T, K, seed=0, slope=0.1):
    net = M.CKNet("S", seed=0)
    with torch.no_grad():                       # non-trivial corrections so the surrogate has gradients
        g = torch.Generator().manual_seed(seed + 1)
        for h in (net.trunk.lon_head, net.trunk.lat_head):
            h[-1].weight.copy_(torch.randn(h[-1].weight.shape, generator=g) * 0.05)
            h[-1].bias.fill_(-0.3)
    bev = torch.randn(T, 256, 50, 100, generator=torch.Generator().manual_seed(seed + 2)).half()
    cand = _cand(T, K, seed=seed)
    return net, net(bev, cand, _status(T), slope=slope), cand


def test_correction_loss_real_gt_finite_and_reaches_heads():
    ref = _ref(GT_TOKENS)
    T, K = 2, 4
    net, o, cand = _student_out(T, K)
    human = _cand(T, 1, noise=0.0)[:, 0]
    idx = L.surrogate_index(ref["ref_gt_ok"], human)
    assert idx.tolist() == [0, 1]
    v0, a0 = o["ego"][0], o["ego"][1]
    b = L.surrogate_batch_k(ref, idx, cand, human, v0, a0)
    assert b["tau0"].shape == (2, K, 8, 3) and b["obj_kf"].shape[1] == int(ref["ref_obj_n"].max())
    loss, means, bad = L.correction_loss(o["corr"]["raw"], b, len(idx), K)
    assert bad == 0 and torch.isfinite(loss) and float(loss.detach()) > 0
    assert set(means) == set(C.SUR_TERMS) | {"P1_minus_P0"}
    loss.backward()
    for h in (net.trunk.lon_head, net.trunk.lat_head):
        assert torch.count_nonzero(h[-1].weight.grad) > 0
    assert net.score_head[-1].weight.grad is None


def test_correction_loss_subset_equals_restricted_batch(monkeypatch):
    ref = _ref(GT_TOKENS)
    T, K = 2, 3
    _, o, cand = _student_out(T, K, seed=3)
    human = _cand(T, 1, noise=0.0)[:, 0]
    v0, a0 = o["ego"][0], o["ego"][1]
    ok1 = ref["ref_gt_ok"].clone()
    ok1[0] = False                                                             # token 0 has no GT
    idx = L.surrogate_index(ok1, human)
    assert idx.tolist() == [1]
    b = L.surrogate_batch_k(ref, idx, cand, human, v0, a0)
    l_sub, m_sub, _ = L.correction_loss(o["corr"]["raw"], b, 1, K)            # flat T*K decode, restricted via idx
    ref1 = {k: v[1:] for k, v in ref.items()}
    raw1 = L.restrict_dec(o["corr"]["raw"], torch.arange(K, 2 * K))
    b1 = L.surrogate_batch_k(ref1, torch.tensor([0]), cand[1:], human[1:], v0[1:], a0[1:])
    l_one, m_one, _ = L.correction_loss(raw1, b1, 1, K)
    torch.testing.assert_close(l_sub, l_one)
    assert m_sub == pytest.approx(m_one)
    z, _, nb = L.correction_loss(o["corr"]["raw"], L.surrogate_batch_k(ref, idx[:0], cand, human, v0, a0), 0, K)
    assert float(z) == 0.0 and nb == 0
    # a non-finite weighted term drops the micro-batch surrogate (E2 rule)
    tr = E.train_refiner_module()
    orig = tr.surrogate_terms_batch

    def bad(*a, **k):
        t = dict(orig(*a, **k))
        t["col"] = t["col"].clone()
        t["col"][0] = float("nan")
        return t
    monkeypatch.setattr(tr, "surrogate_terms_batch", bad)
    l_bad, _, nb = L.correction_loss(o["corr"]["raw"], b, 1, K)
    assert nb == 1 and float(l_bad) == 0.0
