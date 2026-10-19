"""refiner v4 network (navsim/agents/para_ssr/ck/v4): off = CK2 / v3 bit for bit, hazard readout geometry and
numerics, losses, checkpoint round trips, student / teacher entry points.  CPU, synthetic inputs, ~1 min.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_model.py
"""
import json
import math

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import online2 as O
from navsim.agents.para_ssr.ck.cands2 import online_variants
from navsim.agents.para_ssr.ck.model import CKNet, load_ck, save_ck
from navsim.agents.para_ssr.ck.v4 import V4Cfg
from navsim.agents.para_ssr.ck.v4 import losses as V4L
from navsim.agents.para_ssr.ck.v4.readout import noisy_or_logit, pref_loss, time_embed
from navsim.agents.para_ssr.ck.v4.sibling import on_parent
from navsim.agents.para_ssr.refiner.corridor import arrival_time_speed, corridor_geometry
from navsim.agents.para_ssr.refiner.e2e import ego_inputs

T = 2
HAZ = V4Cfg(readout="hazard", pref_head=True)


def _draft(v, kap):
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    s = v * t
    if abs(kap) < 1e-9:
        return torch.stack([s, 0 * s, 0 * s], -1)
    return torch.stack([torch.sin(kap * s) / kap, (1 - torch.cos(kap * s)) / kap, kap * s], -1)


def _pool():
    top = torch.stack([torch.stack([_draft(6 + 0.3 * k, 0.01 * (k - 8) + 1e-4) for k in range(16)])
                       for _ in range(T)])                                           # [T, 16, 8, 3] f64
    V = online_variants(top, torch.full((T,), 6.0, dtype=torch.float64), None)
    status = torch.zeros(T, 8)
    status[:, 1] = 1
    status[:, 4] = 6.0
    return top.float(), V["traj96"].float(), V["valid96"], status


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    top, traj96, valid96, status = _pool()
    return {"top": top, "traj96": traj96, "valid96": valid96, "status": status, "bev": torch.randn(T, 256, 50, 100),
            "par96": top.repeat_interleave(6, 1)}


# ----------------------------------------------------------------------------------------------- config
def test_v4cfg_parse():
    assert not V4Cfg().on and V4Cfg.from_dict(None) == V4Cfg() == V4Cfg.from_dict({})
    c = V4Cfg.from_dict({"readout": "hazard", "pref_head": "true", "horizon": "3.5", "n_freq": "2"})
    assert c.hazard and c.pref_head and c.horizon == 3.5 and c.n_freq == 2 and c.on
    assert V4Cfg.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="unknown"):
        V4Cfg.from_dict({"readout": "hazard", "bogus": 1})
    with pytest.raises(ValueError):
        V4Cfg.from_dict({"readout": "max"})
    with pytest.raises(ValueError, match="temporal"):
        V4Cfg(readout="hazard").validate(temporal_on=True)


# ----------------------------------------------------------------------------------------------- off = CK2
def test_off_is_bit_identical(data):
    n0 = CKNet("S", seed=3)
    n1 = CKNet("S", seed=3, v4=V4Cfg())
    a, b = n0.state_dict(), n1.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    o0 = n0(data["bev"], data["traj96"], data["status"], decode=False)
    o1 = n1(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    for k in ("score_logit", "w_lat", "z_lon", "gate_logit"):
        assert torch.equal(o0[k], o1[k]), k
    # score_set (mean) == trunk.candidates + score
    ego = ego_inputs(data["status"])
    feat, mem = n0.trunk.scene(data["bev"], T)
    o = n0.trunk.candidates(feat, mem, data["traj96"], *ego)
    ss = n0.score_set(feat, mem, ego, data["traj96"], data["par96"])
    assert torch.equal(ss["score_logit"], n0.score(o, T, 96)) and torch.equal(ss["w_lat"], o["w_lat"].float())


def test_v4_modules_do_not_shift_the_ck2_init():
    """v4 heads are built after every CK2 module under their own seed: the shared parameters equal the CK2 init."""
    a = CKNet("S", seed=5).state_dict()
    b = CKNet("S", seed=5, v4=HAZ).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert set(b) - set(a) and all(k.startswith(("v4_readout.", "pref_head.")) for k in set(b) - set(a))


# ----------------------------------------------------------------------------------------------- geometry
def test_on_parent_self_and_siblings(data):
    top, traj96 = data["top"], data["traj96"]
    B = T * 96
    flat = traj96.reshape(B, 8, 3)
    par = data["par96"].reshape(B, 8, 3)
    pg = corridor_geometry(par)
    is_self = (flat == par).flatten(1).all(1)
    assert int(is_self.sum()) == T * 16
    sib = on_parent(pg, flat, is_self)
    # a parent reads its own grid exactly as its corridor's geo channel (same schedule)
    t_ref, _ = arrival_time_speed(pg.S.float(), pg.s.float(), pg.v_end.float())
    assert torch.equal(sib["t"][is_self], t_ref[is_self]) and float(sib["d"][is_self].abs().max()) == 0.0
    v = torch.arange(B) % 6
    t, d = sib["t"], sib["d"]
    reach = t <= 4.0
    # speed siblings stay on the parent path up to the parent's S_8; slower siblings arrive later, a+0.5 earlier.
    # Beyond the parent's S_8 the CK2 a+0.5 variant is extended STRAIGHT (cands2 ext 'straight') while the parent
    # corridor follows its end curvature, so a+0.5 leaves the parent grid there (report 72 I4; up to ~0.4 m here)
    inside = pg.s <= pg.S[:, -2:-1]       # up to the parent's S_7 (the a+0.5 knot 8 already lies beyond its S_8)
    for vi in (1, 2, 3):
        sel = v == vi
        assert float(d[sel][(reach & inside)[sel]].abs().max()) < 0.05
    assert float(d[v == 3][(reach & ~inside)[v == 3]].abs().max()) > 0.1          # the I4 mismatch is visible
    j = 30
    assert bool((t[v == 1, j] > t[v == 0, j] + 0.05).all()) and bool((t[v == 3, j] < t[v == 0, j] - 0.02).all())
    # lateral siblings: the smoothstep ramp (20 % .. 100 % of S_8) ends at ~ -0.5 / +0.5 m, same schedule as the parent
    dk = sib["d_knots"]
    assert abs(float(dk[v == 4, -1].mean()) + 0.5) < 0.06 and abs(float(dk[v == 5, -1].mean()) - 0.5) < 0.06
    assert bool((dk[v == 4, 1:].diff(dim=1) <= 0.02).all()) and bool((dk[v == 5, 1:].diff(dim=1) >= -0.02).all())
    assert float((t[v == 4] - t[v == 0])[reach[v == 0]].abs().mean()) < 0.05


# ----------------------------------------------------------------------------------------------- numerics
def test_noisy_or_and_time_embed():
    a = torch.tensor([[-3.0, -2.0, 0.5, 4.0]])
    m = torch.tensor([[True, True, False, True]])
    p = torch.sigmoid(-a)[m].prod()
    assert abs(float(torch.sigmoid(noisy_or_logit(a, m))) - float(p)) < 1e-6
    big = noisy_or_logit(torch.full((1, 48), -30.0), torch.zeros(1, 48, dtype=torch.bool))
    assert float(big) > 10 and math.isfinite(float(big))              # empty mask: finite (the readout never passes
    # one: read_mask always reads station 0, test_near_stop_reads_and_learns)
    e = time_embed(torch.tensor([0.0, 2.0, 20.0]), 3)
    assert e.shape == (3, 7) and float(e[2, 0]) == 2.0                 # clamped to 8 s


def test_station_targets():
    t = torch.arange(48, dtype=torch.float32)[None, None] * 0.1         # station j reached at 0.1 j s
    m = torch.ones(1, 3, 48, dtype=torch.bool)
    m[..., 41:] = False
    tidx = torch.tensor([[20, -1, -2]])                                  # fail at 2.0 s / pass / unknown
    y = torch.tensor([[0.0, 1.0, 0.0]])
    ok = torch.ones(1, 3, dtype=torch.bool)
    tg, w = V4L.station_targets(t.expand(1, 3, 48), t.expand(1, 3, 48), m, tidx, y, ok)   # zero-length = old rule
    assert float(tg[0, 0, 20]) == 1 and float(w[0, 0, 20]) == 1 and float(tg[0, 0].sum()) == 1
    assert float(w[0, 0, 19]) == 0 and float(w[0, 0, 21]) == 0 and float(w[0, 0, 30]) == 0      # near / censored
    assert float(w[0, 0, :18].sum()) == 18 and float(w[0, 0, 18]) == 0                        # before tau - 0.25 s
    assert float(w[0, 1].sum()) == 41 and float(tg[0, 1].sum()) == 0                           # pass: all read
    assert float(w[0, 2].sum()) == 0                                                           # unknown
    haz = {"a_nc": torch.zeros(1, 3, 48), "a_ttc": torch.zeros(1, 3, 48), "a_dac": torch.zeros(1, 3, 48),
           "haz_mask": m, "t_arr": t.expand(1, 3, 48)}
    y5 = torch.ones(1, 3, 5)
    y5[0, 0] = 0
    l, st = V4L.station_bce(haz, torch.stack([tidx] * 3, -1), y5, ok)
    assert abs(float(l) - math.log(2)) < 1e-5 and st["st_npos_nc"] == 1.0
    assert float(pref_loss(torch.tensor([math.log1p(10.0)]), torch.tensor([10.0]))) < 1e-9
    assert float(pref_loss(torch.tensor([1.0]), torch.tensor([float("nan")]))) == 0.0
    # station intervals: a candidate stopped in bin 0 (only station 0 read, dwells 0..4 s) failing at 3.5 s -> its
    # positive is station 0; a slow one with bins [0, 1.5) [1.5, 3) [3, 4] failing at 3.5 s -> station 2 positive,
    # station 0 negative (left before tau - 0.25 s), station 1 ignored (j* - 1)
    inf = float("inf")
    t_in = torch.full((1, 2, 48), inf)
    t_out = torch.full((1, 2, 48), inf)
    t_in[0, 0, 0], t_out[0, 0, 0] = 0.0, 4.0
    t_in[0, 1, :3], t_out[0, 1, :3] = torch.tensor([0.0, 1.5, 3.0]), torch.tensor([1.5, 3.0, 4.0])
    mm = torch.isfinite(t_in)
    tg, w = V4L.station_targets(t_in, t_out, mm, torch.tensor([[35, 35]]), torch.tensor([[0.0, 0.0]]),
                                torch.ones(1, 2, dtype=torch.bool))
    assert float(tg[0, 0, 0]) == 1 and float(w[0, 0].sum()) == 1
    assert float(tg[0, 1, 2]) == 1 and w[0, 1, :3].tolist() == [1.0, 0.0, 1.0] and float(w[0, 1].sum()) == 2
    # boundary: tau on a bin edge belongs to the later bin ([t_in, t_out)); tau = 4.0 s to the last one
    tg, _ = V4L.station_targets(t_in, t_out, mm, torch.tensor([[40, 15]]), torch.tensor([[0.0, 0.0]]),
                                torch.ones(1, 2, dtype=torch.bool))
    assert float(tg[0, 0, 0]) == 1 and float(tg[0, 1, 1]) == 1


# ----------------------------------------------------------------------------------------------- hazard net
def test_hazard_forward_backward(data):
    net = CKNet("S", seed=0, v4=HAZ)
    net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])
    frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
    assert any(n.startswith("score_head") for n in frozen) and any(n.startswith("trunk.lat_head") for n in frozen)
    o = net(data["bev"], data["traj96"], data["status"], decode=True, cand_parent=data["par96"],
            extra=data["traj96"][:, :8], extra_parent=data["par96"][:, :8])
    assert o["score_logit"].shape == (T, 96, 5) and o["w_lat"].shape == (T, 96, 6) and o["pref_logp"].shape == (T,)
    assert torch.isfinite(o["score_logit"]).all() and float(o["z_lon"].abs().max()) == 0.0
    assert torch.allclose(o["extra_score_logit"], o["score_logit"][:, :8], atol=1e-5)
    p = torch.sigmoid(o["score_logit"])
    assert abs(float(p[..., 2].mean()) - 0.9) < 0.05                   # EP prior
    loss = o["score_logit"].sum() + o["w_lat"].sum() + o["pref_logp"].sum()
    loss.backward()
    nog = [n for n, q in net.named_parameters() if q.requires_grad and q.grad is None]
    assert not nog, nog
    # explicit self parents == no parents
    o2 = net(data["bev"], data["top"], data["status"], decode=False)
    o3 = net(data["bev"], data["top"], data["status"], decode=False, cand_parent=data["top"])
    assert torch.allclose(o2["score_logit"], o3["score_logit"], atol=1e-6)


def test_checkpoint_round_trip(tmp_path, data):
    net = CKNet("S", seed=1, v4=HAZ, corridor_version=2)
    run = tmp_path / "run"
    cfg = {"arm": "S", "seed": 1, "score_hidden": 256, "lead_aux": 0, "corridor_version": 2, "v4": HAZ.to_dict()}
    save_ck(run / "ckpt_last.pt", net, cfg)
    (run / "config.json").write_text(json.dumps(cfg))
    net2, _ = load_ck(run, "last", "cpu")
    assert net2.v4 == HAZ and net2.corridor_version == 2
    a, b = net.state_dict(), net2.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    net.eval()
    o1 = net(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    o2 = net2(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    assert torch.equal(o1["score_logit"], o2["score_logit"])


# ----------------------------------------------------------------------------------------------- student / teacher paths
def test_student_build_and_forward(tmp_path, data):
    cfg = O.CKE2E2Config.from_any({"v4": {"readout": "hazard", "pref_head": True, "corridor_version": 2}})
    stu = O.build_student_ck2(cfg, apply_prior=False)
    assert stu.v4_readout is not None and stu.pref_head is not None and stu.corridor_version == 2
    bev = data["bev"]
    out = O.student_forward2(stu, bev, data["traj96"][:, :48], data["status"], extra=data["traj96"][:, 48:],
                             cand_parent=data["par96"][:, :48], extra_parent=data["par96"][:, 48:])
    assert out["score_logit"].shape == (T, 48, 5) and out["extra_score_logit"].shape == (T, 48, 5)
    assert out["haz"]["a_nc"].shape == (T, 48, 48) and out["extra_haz"]["t_arr"].shape == (T, 48, 48)
    assert out["pref_logp"].shape == (T,) and out["score_all"].shape == (T, 96, 5)
    # same as CKNet.forward over the 96 with parents
    ref = stu(bev, data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    assert torch.allclose(out["score_all"], ref["score_logit"], atol=1e-5)
    # the CK2 student is unchanged by the new code paths (no v4 block)
    s0 = O.build_student_ck2(O.CKE2E2Config.from_any({}), apply_prior=False)
    o0 = O.student_forward2(s0, bev, data["traj96"][:, :48], data["status"], extra=data["traj96"][:, 48:],
                            cand_parent=data["par96"][:, :48])
    assert "haz" not in o0 and "pref_logp" not in o0


def test_teacher_forward_v4(data):
    net = CKNet("S", seed=2, v4=V4Cfg(readout="hazard")).eval()
    ego = ego_inputs(data["status"])
    with torch.no_grad():
        s, w = O.teacher_forward(net, data["bev"], data["traj96"], ego, parents=data["par96"])
        ref = net(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    assert torch.allclose(s, ref["score_logit"], atol=1e-6) and torch.allclose(w, ref["w_lat"], atol=1e-6)


def test_label_set_parents():
    """generation rows: a variant's parent is G_traj[k]; fallback rows: anchors self, variants -> var anchor table."""
    B, KG = 2, 4
    traj = torch.arange(B * KG * 24, dtype=torch.float32).reshape(B, KG, 8, 3)
    G = {"G_traj": traj, "G_vtype": torch.tensor([[0, 0, 1, 4], [0, 0, 3, 2]], dtype=torch.int8),
         "G_col": torch.tensor([[0, 6, 1, 10], [-1, -1, 9, 2]], dtype=torch.int16),
         "has": torch.tensor([True, False])}

    class _S:
        def anchors_t(self, dev):
            return torch.full((256, 8, 3), 7.0) + torch.arange(256.0)[:, None, None]

    targets = {"ck2_var_anchor": torch.tensor([[0] * 16, [3, 9] + [0] * 14])}
    par = O.label_set_parents(G, targets, _S(), "cpu")
    assert torch.equal(par[0, 0], traj[0, 0]) and torch.equal(par[0, 1], traj[0, 1])
    assert torch.equal(par[0, 2], traj[0, 0]) and torch.equal(par[0, 3], traj[0, 1])    # col 1 -> k 0, col 10 -> k 1
    assert torch.equal(par[1, 0], traj[1, 0])                                            # fallback anchor: self
    assert float(par[1, 2, 0, 0]) == 7.0 + 9 and float(par[1, 3, 0, 0]) == 7.0 + 3       # col 9 -> k 1, col 2 -> k 0


def test_arm_sets_and_tags():
    from tools.ck.e2e import launch_util as LU
    from tools.ck.e2e2 import launch_util2 as L2
    base = L2.load_ck2_yaml()
    assert L2.arm_of(base) == "bevkd"
    main = LU.apply_sets(base, LU.read_lines(str(L2.MAIN_SET)))
    assert L2.arm_of(main) == "main" and main["bev_kd"]["enabled"] is False
    v4 = LU.apply_sets(base, LU.read_lines(str(L2.V4_SET)))
    assert L2.arm_of(v4) == "v4" and v4["bev_kd"]["enabled"] is True
    c = O.CKE2E2Config.from_any(dict(v4, enabled=False))
    assert c.v4_net().hazard and c.v4["corridor_version"] == 2 and c.v4["kd_parent_only"]


# ----------------------------------------------------------------------------------------------- evidence readout
EVI = V4Cfg(readout="evidence", pref_head=True)


def test_grid_read_and_points_in_boxes():
    from navsim.agents.para_ssr.ck.v4.evidence import grid_read
    from navsim.agents.para_ssr.ck.v4.losses import _points_in_boxes
    G = torch.arange(48 * 17, dtype=torch.float32).reshape(1, 48, 17, 1)
    s_st = ((torch.arange(48) + 0.5) * 1.0)[None]
    # station 10 (s = 10.5), lateral offset index 8 (d = 0) -> the cell value exactly; half way -> the mean
    v, ok = grid_read(G, s_st, torch.tensor([[10.5, 11.0, 200.0]]), torch.tensor([[0.0, 0.3, 0.0]]))
    assert bool(ok[0, 0]) and float(v[0, 0, 0]) == 10 * 17 + 8
    assert abs(float(v[0, 1, 0]) - 0.25 * (G[0, 10, 8, 0] + G[0, 11, 8, 0] + G[0, 10, 9, 0] + G[0, 11, 9, 0])) < 1e-4
    assert not bool(ok[0, 2]) and float(v[0, 2, 0]) == 0.0
    p = torch.tensor([[[5.0, 0.0], [5.0, 1.5], [9.0, 0.0]]], dtype=torch.float64)
    boxes = torch.tensor([[[[5.0, 0.0, 0.0, 4.0, 2.0]], [[5.0, 0.0, math.pi / 2, 4.0, 2.0]]]], dtype=torch.float64)
    obs = torch.tensor([[[True], [False]]])
    occ = _points_in_boxes(p, boxes, obs)
    assert occ.shape == (1, 3, 1) and occ[0, :, 0].tolist() == [True, False, False]
    occ2 = _points_in_boxes(p, boxes, torch.tensor([[[False], [True]]]))          # rotated box covers (5, 1.5)
    assert occ2[0, :, 0].tolist() == [True, True, False]


def test_evidence_forward_backward_and_aux(data):
    net = CKNet("S", seed=0, v4=EVI)
    net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])
    o = net(data["bev"], data["traj96"], data["status"], decode=False, cand_parent=data["par96"])
    h = o["haz"]
    assert o["score_logit"].shape == (T, 96, 5) and torch.isfinite(o["score_logit"]).all()
    assert h["occ_logit"].shape == (T, 96, 48, 17, 10) and h["margin"].shape == (T, 96, 48, 17)   # bins to 5.0 s
    assert h["pts"].shape == (T, 96, 48, 17, 2) and int(h["is_self"].sum()) == T * 16
    p = torch.sigmoid(o["score_logit"])
    assert float(p[..., 0].min()) > 0.5                          # sparse occupancy prior -> NC mostly passes
    # siblings read the same scene field as their parent (identical parent pass)
    assert torch.equal(h["occ_logit"][:, 1], h["occ_logit"][:, 0]) and torch.equal(h["margin"][:, 5], h["margin"][:, 0])
    # aux targets from a synthetic GT scene: one parked box 6 m ahead on the parent path, a flat +3 m SDF
    A = 2
    kf = torch.zeros(T, A, 11, 6)
    kf[:, 0, :, 0], kf[:, 0, :, 5] = 6.0, 1.0
    first = torch.zeros(T, A, 6)
    first[:, :, 0], first[:, :, 1] = 4.6, 1.9                         # L, W (gt_future FI_L / FI_W = 0 / 1)
    meta = torch.zeros(T, A, 5, dtype=torch.long)
    meta[:, :, 3] = 10                                                # MT_LAST: last keyframe (MT_FIRST 0)
    ref = {"ref_obj_kf": kf, "ref_obj_first": first, "ref_obj_meta": meta, "ref_obj_n": torch.tensor([1, 1]),
           "ref_obj_n_kf": torch.tensor([11, 11]), "ref_sdf": torch.full((T, 320, 256), 3.0, dtype=torch.float16),
           "ref_gt_ok": torch.tensor([True, False])}
    l_occ, l_m, st = V4L.evidence_aux(h, ref, max_parents=2)
    assert st["ev_n"] == 2.0 and st["ev_occ_rate"] > 0 and torch.isfinite(l_occ) and torch.isfinite(l_m)
    assert st["ev_sdf_valid"] > 0.9                                # strongly curved test drafts leave the E grid
    (o["score_logit"].sum() + o["w_lat"].sum() + o["pref_logp"].sum() + l_occ + l_m).backward()
    nog = [n for n, q in net.named_parameters() if q.requires_grad and q.grad is None]
    assert not nog, nog


# ----------------------------------------------------------------------------------------------- corrector A
def test_corrector_identity_credit_awr(data):
    from navsim.agents.para_ssr.ck.v4 import corrector as CR
    net = CKNet("S", seed=0, v4=V4Cfg(readout="hazard", pref_head=True, corrector=True, corr_k=4))
    ego = ego_inputs(data["status"])
    feat, mem = net.trunk.scene(data["bev"], T)
    pref = net.scene_pref(mem, ego)
    po = net.score_set(feat, mem, ego, data["traj96"], data["par96"], pref=pref, return_ctx=True)
    assert po["ctx"].shape == (T, 96, 3 * 192)
    ce = net.correct_topk(feat, mem, ego, data["traj96"], data["par96"], po, pref=pref)
    assert ce["src_idx"].shape == (T, 4) and ce["edit_traj"].shape == (T, 4, 8, 3)
    # mu = 0 at init -> the edit is its source (decoder mode B anchoring), re-scored like the source
    src = torch.gather(data["traj96"], 1, ce["src_idx"][..., None, None].expand(T, 4, 8, 3))
    assert float((ce["edit_traj"] - src).abs().max()) < 1e-4 and float(ce["mu"].abs().max()) == 0.0
    sc = CR.critic_select_score(po["score_logit"])
    assert torch.equal(ce["src_idx"], torch.topk(sc, 4, 1).indices)
    # sampled edits move; their log-prob under the current policy is recomputed exactly by corrector_logp
    ce2 = net.correct_topk(feat, mem, ego, data["traj96"], data["par96"], po, pref=pref, sample=True,
                           generator=torch.Generator().manual_seed(0))
    assert float((ce2["edit_traj"] - src).abs().max()) > 1e-3
    lp = net.corrector_logp(feat, mem, ego, src, torch.gather(data["par96"], 1, ce["src_idx"][..., None, None]
                                                              .expand(T, 4, 8, 3)), ce2["u"], pref=pref)
    assert torch.allclose(lp, ce2["logp"], atol=1e-4)
    # credit: an edit that is picked and better than its source gets A > 0; a picked trap A < 0 (exact
    # leave-one-out values: f(S \\ edit) is the softmax over the two others = 0.35 / 0.7)
    s_all = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    R_all = torch.tensor([[0.5, 0.2, 1.0], [0.5, 0.9, 0.0]])
    ok = torch.ones(2, 3, dtype=torch.bool)
    c = CR.credit(s_all, R_all, ok, torch.tensor([[2], [2]]), torch.tensor([[0], [0]]))
    assert abs(float(c["A_sel"][0, 0]) - 0.65) < 1e-6 and abs(float(c["A_sel"][1, 0]) + 0.7) < 1e-6
    assert float(c["A"][0, 0]) > 0 and float(c["A"][1, 0]) < 0 and abs(float(c["A_loc"][0, 0]) - 0.5) < 1e-6
    # AWR: weights favour positive advantages; gradient reaches mu / log_sig
    l = CR.awr_loss(ce2["logp"], torch.randn(T, 4) * 0.1, torch.ones(T, 4, dtype=torch.bool))
    l.backward()
    assert net.corrector.mu.weight.grad is not None and float(net.corrector.mu.weight.grad.abs().sum()) > 0
