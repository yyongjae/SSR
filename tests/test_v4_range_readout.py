"""refiner v4_3 M4 readout 'range' (navsim/agents/para_ssr/ck/v4/range_readout.py) + the v4_3 config: config versions
(v4_2 dicts and the trained v4T10_haz / v4M10_haz teachers load), off / v4_2 readouts untouched, parameter budget
(arms S / T / M, every range option), shapes / gradients / dedupe and pctx re-use, the official ego state (status[4],
status[6]) for the LQR states, slacks and critical points equal to range_targets.executed_slack on the same fields, the
TTC B / U channel rule, EP evidence detached, the vmax feature, CPU fp16 autocast.  CPU, synthetic inputs, ~2 min.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_range_readout.py
"""
import json
from pathlib import Path

import pytest
import torch

from navsim.agents.para_ssr.ck.cands2 import online_variants
from navsim.agents.para_ssr.ck.model import CKNet, load_ck
from navsim.agents.para_ssr.ck.v4 import V4Cfg
from navsim.agents.para_ssr.ck.v4 import config as C
from navsim.agents.para_ssr.ck.v4 import lqr as LQR
from navsim.agents.para_ssr.ck.v4 import range_readout as RR
from navsim.agents.para_ssr.ck.v4 import range_targets as RT
from navsim.agents.para_ssr.ck.v4.readout import I_DAC, I_EP, I_NC, I_TTC
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry, lateral_offsets
from navsim.agents.para_ssr.refiner.e2e import ego_inputs

RNG = V4Cfg.range_arm()
TEACHERS = Path("/home/external-user/ssd/yongjae_refiner/ck/v4/train")
T = 2


def _draft(v, kap=0.0, dtype=torch.float32):
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    s = v * t
    if abs(kap) < 1e-9:
        d = torch.stack([s, 0 * s, 0 * s], -1)
    else:
        d = torch.stack([torch.sin(kap * s) / kap, (1 - torch.cos(kap * s)) / kap, kap * s], -1)
    return d.to(dtype)


def _status(v0, vy=0.0, ax=0.0):
    v0 = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    s = torch.zeros(v0.numel(), 8)
    s[:, 1] = 1
    s[:, 4], s[:, 5], s[:, 6] = v0, vy, ax
    return s


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    top = torch.stack([torch.stack([_draft(6 + 0.3 * k, 0.01 * (k - 8) + 1e-4, torch.float64) for k in range(16)])
                       for _ in range(T)])
    V = online_variants(top, torch.full((T,), 6.0, dtype=torch.float64), None)
    return {"top": top.float(), "traj96": V["traj96"].float(), "par96": top.float().repeat_interleave(6, 1),
            "status": _status([6.0, 6.0], vy=0.2, ax=-0.3), "bev": torch.randn(T, 256, 50, 100)}


def _trainable(net) -> int:
    frozen = ("trunk.lon_head", "trunk.gate_head")          # CK2 as trained (train_ck2.FROZEN_MODULES, online2)
    return sum(p.numel() for n, p in net.named_parameters() if p.requires_grad and not n.startswith(frozen))


# ----------------------------------------------------------------------------------------------- config
def test_config_v43_versions_and_rules():
    assert C.CONFIG_VERSION == "v4_3" and V4Cfg().version == "v4_3" and not V4Cfg().on
    d = V4Cfg()
    assert (d.exec_model, d.ep_evidence, d.vmax, d.edits, d.ev_tbins, d.ev_tbin_s) == ("plan", False, "off", "off",
                                                                                       10, 0.25)
    r = V4Cfg.range_arm()
    assert r.range and r.parent_pass and r.exec_model == "lqr" and r.ev_tbins == 20 and r.ep_evidence and r.pref_head
    assert V4Cfg.from_dict(r.to_dict()) == r
    assert V4Cfg.from_dict(V4Cfg.range_arm(edits="computed").to_dict()).computed_edits
    with pytest.raises(ValueError, match="ev_tbins x ev_tbin_s"):        # the default 10 bins of 0.25 s: 2.5 s only
        V4Cfg(readout="range").validate()
    assert V4Cfg(readout="range", ev_tbins=10, ev_tbin_s=0.5).validate().range
    with pytest.raises(ValueError):
        V4Cfg.range_arm(ev_tbins=21)                                      # cap 20
    with pytest.raises(ValueError):
        V4Cfg(readout="evidence", ev_tbins=11).validate()                 # evidence keeps its 10 x 0.5 s cap
    for kw in ({"ep_evidence": True}, {"exec_model": "lqr"}, {"vmax": "feature"}, {"edits": "computed"},
               {"ev_tbin_s": 0.5}, {"edit_k": 4}, {"edit_dt": 0.4}, {"edit_dlat": 0.5}, {"rg_hidden": 64},
               {"ep_ev_hidden": 70}):
        for ro in ("mean", "hazard", "evidence"):         # every v4_3 field (V43_KEYS) is range-only
            with pytest.raises(ValueError, match="only apply to readout 'range'"):
                V4Cfg(readout=ro, **kw).validate()
        assert V4Cfg.range_arm(**kw, **({"ev_tbins": 10} if "ev_tbin_s" in kw else {})).range   # accepted for 'range'
    for kw in ({"exec_model": "x"}, {"vmax": "on"}, {"edits": "learned"}, {"edit_k": 0}, {"edit_dt": -1.0}):
        with pytest.raises(ValueError):
            V4Cfg.range_arm(**kw)
    # v4_2 dicts (no v4_3 key) load as the current version; equal to a fresh config (train_ck2 resume check)
    v42 = {k: v for k, v in V4Cfg(readout="hazard", pref_head=True).to_dict().items() if k not in C.V43_KEYS}
    v42["version"] = "v4_2"
    c = V4Cfg.from_dict(v42)
    assert c == V4Cfg(readout="hazard", pref_head=True) and c.version == "v4_3"
    with pytest.raises(ValueError, match="v4_2"):
        V4Cfg.from_dict(dict(v42, ep_evidence=False))
    with pytest.raises(ValueError, match="version"):
        V4Cfg.from_dict(dict(v42, version="v4_1"))
    with pytest.raises(ValueError):
        V4Cfg(readout="hazard", version="v4_2").validate()                # objects are always the current version


@pytest.mark.parametrize("run", ["v4T10_haz", "v4M10_haz"])
def test_trained_v42_teachers_load(run):
    d = TEACHERS / run
    if not (d / "config.json").is_file() or not (d / "ckpt_last.pt").is_file():
        pytest.skip(f"{d} not available")
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["v4"]["version"] == "v4_2"
    net, _ = load_ck(d, "last", "cpu")                                    # strict state-dict load
    assert net.v4.hazard and net.v4.version == "v4_3" and net.v4 == V4Cfg.from_dict(cfg["v4"])
    assert net.v4 == V4Cfg(readout="hazard", pref_head=True)          # = a fresh config (train_ck2 resume check)


def test_off_and_v42_readouts_untouched():
    """V4Cfg() / hazard / evidence build no range module; their state dicts keep the v4_2 keys (bit-identity of the
    outputs vs the v4_2 code: tests/test_v4_model.py::test_off_is_bit_identical and the v4_2 tests; checked against the
    stage-1 commit on 1,364 tensors, see the M4 report)."""
    a = CKNet("S", seed=5, v4=V4Cfg(readout="hazard", pref_head=True)).state_dict()
    b = CKNet("S", seed=5, v4=RNG).state_dict()
    ck2 = CKNet("S", seed=5).state_dict()
    assert all(torch.equal(ck2[k], b[k]) for k in ck2)                  # the range heads do not shift the CK2 init
    assert all(k.startswith(("v4_readout.", "pref_head.")) for k in set(b) - set(ck2))
    assert not any(k.startswith("v4_readout.blk_head") for k in a) and "v4_readout.h_nc.0.weight" in a


# ----------------------------------------------------------------------------------------------- budget
@pytest.mark.parametrize("arm,ck2", [("S", 3_391_755), ("T", 3_391_243), ("M", 3_391_243)])
def test_param_budget_range(arm, ck2):
    assert _trainable(CKNet(arm, seed=0)) == ck2
    for kw in ({}, {"ep_evidence": False}, {"exec_model": "plan"}, {"vmax": "feature"}, {"edits": "computed"},
               {"exec_model": "plan", "ep_evidence": False, "vmax": "feature"}):
        n = _trainable(CKNet(arm, seed=0, v4=V4Cfg.range_arm(**kw)))
        assert abs(n - ck2) / ck2 < 0.002, (arm, kw, n, ck2)
    na = _trainable(CKNet(arm, seed=0, v4=V4Cfg.range_arm(corrector=True)))
    assert na > ck2 + 100_000                                            # the corrector A is outside the critic budget


# ----------------------------------------------------------------------------------------------- forward
def test_range_forward_shapes_grad_and_priors(data):
    net = CKNet("S", seed=0, v4=V4Cfg.range_arm(vmax="feature"))
    net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])
    o = net(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"],
            extra=data["traj96"][:, :8], extra_parent=data["par96"][:, :8])
    h = o["haz"]
    assert o["score_logit"].shape == (T, 96, 5) and torch.isfinite(o["score_logit"]).all()
    assert h["occ_logit"].shape == (T, 96, 48, 17, 20) and h["margin"].shape == (T, 96, 48, 17)
    assert h["occ_logit_u"].shape == (T, 16, 48, 17, 20) and h["ovl_logit_u"].shape == (T, 16, 48, 17, 20)
    assert h["margin_u"].shape == (T, 16, 48, 17) and h["exec_states"].shape == (T, 96, 41, 6)
    for k in ("lp_nc", "lp_ttc", "lp_dac", "slack_nc", "slack_ttc", "slack_dac", "exec_ratio", "overspeed"):
        assert h[k].shape == (T, 96) and torch.isfinite(h[k]).all(), k
    assert h["crit_station"].shape == (T, 96, 3) and h["crit_time"].shape == (T, 96, 3)
    assert h["crit_chan"].shape == (T, 96)
    assert bool((h["lp_nc"] <= 0).all()) and int(h["is_self"].sum()) == T * 16
    # each candidate reads its parent's row: the per-candidate fields are the parent's
    pi = h["parent_index"]
    assert torch.equal(h["occ_logit"][0, 7], h["occ_logit_u"][0, pi[0, 7]]) and int(pi[0, 7]) == 1
    assert torch.equal(h["margin"][1, 95], h["margin_u"][1, 15])
    assert torch.allclose(o["extra_score_logit"], o["score_logit"][:, :8], atol=1e-5)
    p = torch.sigmoid(o["score_logit"]).mean((0, 1))
    assert abs(float(p[I_EP]) - 0.9) < 0.05 and float(p[I_NC]) > 0.9 and float(p[I_DAC]) > 0.9
    (o["score_logit"].sum() + o["w_lat"].sum() + o["pref_logp"].sum()).backward()
    nog = [n for n, q in net.named_parameters() if q.requires_grad and q.grad is None]
    assert not nog, nog
    rd = net.v4_readout
    for mod in (rd.blk_head, rd.marg_head, rd.ep_head, rd.comfort_head, rd.lat_head, rd.resid):
        assert float(mod[-1].weight.grad.abs().sum()) > 0
    assert float(rd.log_eps.grad.abs()) > 0
    U = rd.blk_head[-1].weight.grad[rd.M:]                               # the U channel is read (TTC, 6 m/s)
    assert float(U.abs().sum()) > 0


def test_lqr_states_use_the_official_ego_state(data):
    """exec_states = lqr.simulate from (status[..., 4], status[..., 6]) -- NOT the hypot(vx, vy) of ego_inputs."""
    net = CKNet("S", seed=0, v4=RNG).eval()
    st = data["status"]
    with torch.no_grad():
        o = net(data["bev"], data["traj96"], st, decode=False, cand_parent=data["par96"])
    es = o["haz"]["exec_states"]
    assert torch.equal(es[:, :, 0, 3], st[:, 4:5].expand(T, 96))
    assert torch.equal(es[:, :, 0, 4], st[:, 6:7].expand(T, 96))
    ref = LQR.simulate(data["traj96"], st[:, None, [4, 6]], dtype=torch.float32)
    assert torch.equal(es, ref)
    sp, tp = RR.candidate_states(data["traj96"][0], st[:1, 4].expand(96), st[:1, 6].expand(96), "plan")
    assert sp.shape == (96, 9, 6) and float(tp[-1]) == 4.0 and torch.equal(sp[:, 1:, :3], data["traj96"][0])


def test_dedupe_and_pctx_reuse_bit_identical(data):
    net = CKNet("S", seed=1, v4=RNG).eval()
    ego = ego_inputs(data["status"])
    with torch.no_grad():
        feat, mem = net.trunk.scene(data["bev"], T)
        pref = net.scene_pref(mem, ego)
        a = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_pctx=True)
        b = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, dedupe=False)
        cols = torch.tensor([[5, 0, 17, 95], [3, 44, 2, 60]])
        g = lambda x: torch.gather(x, 1, cols.reshape(T, 4, *([1] * (x.dim() - 2))).expand(T, 4, *x.shape[2:]))
        c = net.score_set(feat, mem, ego, g(data["traj96"]), pref=pref,
                          parent_index=torch.gather(a["parent_index"], 1, cols), pctx=a["pctx"])
    assert "rg_fields" in a["pctx"] and a["pctx"]["rg_fields"]["blk"].shape[0] == T * 16
    assert torch.equal(a["score_logit"], b["score_logit"]) and torch.equal(a["w_lat"], b["w_lat"])
    for k in ("occ_logit", "margin", "lp_nc", "lp_ttc", "lp_dac", "slack_nc", "slack_ttc", "slack_dac", "crit_station",
              "crit_time", "exec_states"):
        assert torch.equal(a["haz"][k], b["haz"][k]), k
    assert torch.equal(c["score_logit"], g(a["score_logit"]))
    assert torch.equal(c["haz"]["slack_ttc"], g(a["haz"]["slack_ttc"]))


# ----------------------------------------------------------------------------------------------- rules on known fields
def _patched(net, blk_fn, D_fn):
    """replace the learned scene fields by known ones (every parent row gets blk_fn(R), D_fn(R))."""
    net.v4_readout.scene = lambda xs, Hc: {"blk": blk_fn(xs.shape[0]), "margin": D_fn(xs.shape[0])}
    return net


def _scene(net, bev, cand, par, status):
    ego = ego_inputs(status)
    feat, mem = net.trunk.scene(bev, cand.shape[0])
    return net.score_set(feat, mem, ego, cand, par, pref=net.scene_pref(mem, ego))


def test_slacks_equal_range_targets_executed_slack():
    """hard slack_nc / slack_ttc / crit points == range_targets.executed_slack on B (U = B), DAC on the same D grid."""
    torch.manual_seed(0)
    net = CKNet("S", seed=0, v4=RNG).eval()
    par = _draft(8.0, 0.02)
    cand = torch.stack([par, _draft(3.0, 0.02), _draft(8.0, 0.02), _draft(11.0, 0.02)])
    cand[2, :, 1] += torch.linspace(0.3, 2.6, 8)                          # drifts left over the curb
    cand = cand[None]
    pars = par[None, None].expand(1, 4, 8, 3)
    geom = corridor_geometry(par[None], version=1)
    s = geom.s[0]
    d = lateral_offsets()
    Bb = torch.zeros(1, 48, 17, 20, dtype=torch.bool)
    tb = torch.arange(20) * 0.25
    Bb[0] = (((s >= 14) & (s < 24))[:, None, None] & (d.abs() <= 1.3)[None, :, None]
             & ((tb >= 1.5) & (tb < 3.0))[None, None])
    Dg = (2.6 - d.abs())[None, None].expand(1, 48, 17).clone()           # curbs at +-2.6 m
    blk = torch.cat([torch.where(Bb, 6.0, -6.0)] * 2, -1)
    _patched(net, lambda R: blk.expand(R, *blk.shape[1:]).contiguous(), lambda R: Dg.expand(R, 48, 17).contiguous())
    with torch.no_grad():
        o = _scene(net, torch.randn(1, 256, 50, 100), cand, pars, _status(8.0))
    h = o["haz"]
    fr = {k: v.float() for k, v in RT.parent_frame(geom).items()}
    ref = RT.executed_slack(h["exec_states"][0], fr, Bb, field=Dg, index=torch.zeros(4, dtype=torch.long))
    assert torch.equal(h["slack_nc"][0], ref["slack_nc"]) and torch.equal(h["slack_ttc"][0], ref["slack_ttc"])
    assert torch.equal(h["crit_station"][0, :, :2], ref["crit_station"][:, :2])
    assert torch.equal(h["crit_time"][0, :, :2], ref["crit_time"][:, :2])
    assert torch.allclose(h["slack_dac"][0], ref["slack_dac"], atol=1e-4)
    hit_nc, hit_ttc, hit_dac = ref["hit_nc"], ref["hit_ttc"], ref["hit_dac"]
    assert bool(hit_nc[0]) and bool(hit_nc[3]) and not bool(hit_nc[1]) and bool(hit_dac[2]) and not bool(hit_dac[0])
    # soft evidence agrees with the hard reads: hits fail (P_pass < 0.5), the free candidate passes
    p = torch.sigmoid(o["score_logit"][0])
    assert bool((p[hit_nc, I_NC] < 0.5).all()) and float(p[1, I_NC]) > 0.95
    assert float(p[2, I_DAC]) < 0.5 and float(p[0, I_DAC]) > 0.6       # soft max over 164 corners (residual calibrates)


def test_ttc_channel_rule():
    """B empty, U blocked everywhere: NC never fails; TTC fails only where some read has |v| delta >= 2 m (fast ego),
    not for a slow one (max |v| delta = 0.9 m)."""
    net = CKNet("S", seed=0, v4=RNG).eval()
    M = 20
    blk = torch.cat([torch.full((1, 48, 17, M), -12.0), torch.full((1, 48, 17, M), 6.0)], -1)
    _patched(net, lambda R: blk.expand(R, *blk.shape[1:]).contiguous(), lambda R: torch.full((R, 48, 17), 3.0))
    cand = torch.stack([_draft(8.0)[None], _draft(1.0)[None]])               # token 0 fast, token 1 slow
    with torch.no_grad():
        o = _scene(net, torch.randn(2, 256, 50, 100), cand, cand, _status([8.0, 1.0]))
    h = o["haz"]
    assert bool((h["slack_nc"] > 0).all())
    assert float(h["slack_ttc"][0, 0]) < 0 and int(h["crit_chan"][0, 0]) == 1
    assert float(h["slack_ttc"][1, 0]) > 0
    p = torch.sigmoid(o["score_logit"])
    assert float(p[0, 0, I_TTC]) < 0.5 and float(p[1, 0, I_TTC]) > 0.99 and float(p[0, 0, I_NC]) > 0.99


def test_ep_evidence_detached_and_lateral_weights():
    net = CKNet("S", seed=0, v4=RNG)
    blk = torch.full((1, 48, 17, 40), -2.0, requires_grad=True)
    Dg = torch.full((1, 48, 17), 1.5, requires_grad=True)
    _patched(net, lambda R: blk.expand(R, 48, 17, 40), lambda R: Dg.expand(R, 48, 17))
    cand = torch.stack([_draft(6.0), _draft(7.0)])[None]
    o = _scene(net, torch.randn(1, 256, 50, 100), cand, cand[:, :1].expand(1, 2, 8, 3), _status(6.0))
    ge = torch.autograd.grad(o["score_logit"][..., I_EP].sum(), [blk, Dg], allow_unused=True, retain_graph=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in ge)    # the EP head reads detached evidence
    gn = torch.autograd.grad(o["score_logit"][..., I_NC].sum(), [blk], allow_unused=True, retain_graph=True)[0]
    gd = torch.autograd.grad(o["score_logit"][..., I_DAC].sum(), [Dg], allow_unused=True)[0]
    assert gn is not None and float(gn.abs().sum()) > 0 and gd is not None and float(gd.abs().sum()) > 0


def test_grid_margins_and_vmax_feature():
    # straight road with curbs at +-3 m: grid_margins == range_targets.margins (box half width 1.1485)
    geom = corridor_geometry(_draft(8.0)[None])
    d = lateral_offsets()
    Dg = (3.0 - d.abs())[None, None].expand(1, 48, 17).contiguous()
    mg = RR.grid_margins(Dg, torch.clamp(geom.S_look / 48, min=1.0))
    mr = RT.margins(Dg, geom)
    assert torch.allclose(mg["m_left"], torch.full_like(mg["m_left"], 3.0 - 1.1485), atol=1e-4)
    assert torch.allclose(mg["m_left"], mr["m_left"], atol=0.02)
    assert torch.allclose(mg["m_right"], mr["m_right"], atol=0.02)
    # curved parent at 12 m/s: narrow road -> v_max ~ 1 m/s, overspeed; wide road -> none; vmax off -> 0
    out = {}
    for name, half, mode in (("narrow", 1.6, "feature"), ("wide", 4.8, "feature"), ("off", 1.6, "off")):
        net = CKNet("S", seed=0, v4=V4Cfg.range_arm(vmax=mode)).eval()
        Dn = (half - d.abs())[None, None].expand(1, 48, 17).contiguous()
        _patched(net, lambda R: torch.full((R, 48, 17, 40), -6.0), lambda R, Dn=Dn: Dn.expand(R, 48, 17).contiguous())
        c = _draft(12.0, 0.05)[None, None]
        with torch.no_grad():
            out[name] = float(_scene(net, torch.randn(1, 256, 50, 100), c, c, _status(12.0))["haz"]["overspeed"][0, 0])
    assert out["narrow"] > 5.0 and out["wide"] == 0.0 and out["off"] == 0.0


def test_cpu_fp16_autocast_forward(data):
    net = CKNet("S", seed=0, v4=RNG)
    with torch.autocast("cpu", dtype=torch.float16):
        o = net(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    assert o["score_logit"].dtype == torch.float32 and torch.isfinite(o["score_logit"]).all()
    o["score_logit"].sum().backward()
    assert net.v4_readout.blk_head[0].weight.grad is not None


def test_corrector_topk_on_the_range_readout(data):
    """corrector A on the range critic: identity at init, edits re-scored through the pool's pctx (rg_fields)."""
    net = CKNet("S", seed=0, v4=V4Cfg.range_arm(corrector=True, corr_k=4)).eval()
    ego = ego_inputs(data["status"])
    with torch.no_grad():
        feat, mem = net.trunk.scene(data["bev"], T)
        pref = net.scene_pref(mem, ego)
        po = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_ctx=True,
                           return_pctx=True)
        ce = net.correct_topk(feat, mem, ego, data["traj96"], data["par96"], po, pref=pref)
    g = lambda x: torch.gather(x, 1, ce["src_idx"].reshape(T, 4, *([1] * (x.dim() - 2))).expand(T, 4, *x.shape[2:]))
    assert float((ce["edit_traj"] - g(data["traj96"])).abs().max()) < 1e-4
    assert torch.allclose(ce["edit_score_logit"], g(po["score_logit"]), atol=1e-2)
    assert ce["edit_haz"]["exec_states"].shape == (T, 4, 41, 6)
