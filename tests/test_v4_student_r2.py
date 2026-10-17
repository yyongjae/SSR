"""refiner v4_3 student with the Phase 0b R2 representation (docs/refiner_v4/V4_3_SPEC.md 'Phase 0b' / 'Phase 1c'):
the student arm tools/ck/e2e2/v4_3_arm.set is R2 (rg_read any4 + rg_stopped_aware) to match the R2 teachers
v4T10_rng2 / v4M10_rng2.

  (a) computed edits (ck/v4/edits.py) read the fields with the net's rules: synthetic fields where the any4 read (OR of
      the 2 x 2 cells) or the Bp group (TTC's B read below the 2 m switch) changes the edit; the any4 target checked
      independently (range_targets.executed_slack(read='any4') critical state + range_targets._cells any4 + the OR
      window); any4_targets = hard_evidence's critical reads on random fields; with Bp = B the R2 edits equal the R1
      edits (the channel rule only re-routes TTC's B read);
  (b) V4Cfg accepts edits 'computed' with rg_read any4 / rg_stopped_aware (validate only: same network);
  (c) online2: kd_constraint.occ_channels B / BU / BBp / all, the teachers' requested field groups (teacher_forward /
      TeacherPair2.run), constraint_kd per group, occ_channels with Bp refused for an R1 student, the runtime check
      (grid and rg_stopped_aware of both teachers = the student's; rg_read may differ: the recorded decision of
      tests/test_v4_r2.py::test_constraint_teachers_b_semantics_check);
  (d) launch_util2.kd_constraint_check: the same rule and message before launch;
  (e) + (f) the R2 arm with R2 teachers passes the runtime and launch checks and runs one CPU loss step (warm-up, and
      on-policy with computed edits recorded, v4/edit_clear_frac logged) on the real 2-token fixture; an R1 student
      with R2 teachers (and the R2 arm with R1 teachers) is refused at launch and at runtime; a student whose rg_read
      differs from the teachers' (same rg_stopped_aware) is accepted at both, rg_read_differs recorded.
CPU only:

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_student_r2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
import test_ck2_e2e_loss as TLOSS  # noqa: E402
import test_v4_edits as TE  # noqa: E402
import test_v4_wiring as TW  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.model import CKNet  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import V4Cfg  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import edits as ED  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import lqr as LQR  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import range_targets as RT  # noqa: E402
from navsim.agents.para_ssr.ck.v4.range_readout import CH_BP, CH_U, hard_evidence, read_geometry  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry, lateral_offsets  # noqa: E402

M = 20
TB = torch.arange(M) * 0.25
R2 = dict(rg_read="any4", rg_stopped_aware=True)
V43_R2 = dict(TW.V43, rg_read="any4", rg_stopped_aware=True)          # the v4_3 arm's student v4 block (R2)
have = TW.have


def _net(**kw):
    torch.manual_seed(0)
    return CKNet("S", seed=0, v4=V4Cfg.range_arm(edits="computed", **kw)).eval()


def _cells_where(par, s0, s1, t0, t1, lat):
    """[48, 17, M] bool: stations s in [s0, s1), lateral offsets with lat(d), bins in [t0, t1) on par's grid."""
    s, d = corridor_geometry(par[None]).s[0], lateral_offsets()
    return ((s >= s0) & (s < s1))[:, None, None] & lat(d)[None, :, None] & ((TB >= t0) & (TB < t1))[None, None]


def _fields(par, groups):
    """known fields: blk [1, 48, 17, G M] (+-8 logits of the bool groups), wide-road D, the parent's geometry."""
    blk = torch.cat([torch.where(g, 8.0, -8.0) for g in groups], -1)[None]
    _, D, geom = TE._fields(par)
    return blk, D, geom


def _sorted(ce, keys=("edit_traj", "a_lon", "d_lat", "edit_kind", "edit_need", "edit_clear", "u_rule")):
    o = torch.argsort(ce["src_idx"], 1)
    return {k: torch.gather(ce[k], 1, o.reshape(*o.shape, *([1] * (ce[k].dim() - 2))).expand_as(ce[k]))
            for k in keys + ("src_idx",)}


# ----------------------------------------------------------------------------------------------- (b) config
def test_config_edits_with_r2_accepted_same_network():
    a = V4Cfg.range_arm(edits="computed", **R2)
    assert a.computed_edits and a.rg_read == "any4" and a.rg_stopped_aware
    assert V4Cfg.from_dict(a.to_dict()) == a
    for kw in ({"rg_read": "any4"}, {"rg_stopped_aware": True}):
        assert V4Cfg.range_arm(edits="computed", **kw).validate().computed_edits
    # validate only: edits does not change the network (state dict keys / shapes / seeded values)
    s0 = CKNet("S", seed=0, v4=V4Cfg.range_arm(**R2)).state_dict()
    s1 = CKNet("S", seed=0, v4=a).state_dict()
    assert s0.keys() == s1.keys() and all(torch.equal(s0[k], s1[k]) for k in s0)


# ----------------------------------------------------------------------------------------------- (a) edits
def _any4_target(src, v, geom, Bf, key):
    """independent any4 longitudinal target of src [8, 3] (executed at v) on the bool field Bf [48, 17, M] (B = U): the
    critical state of range_targets.executed_slack(read='any4', per_state=True), its read point's 4 cells
    (range_targets._cells 'any4'), the OR window and the cut max(0, (jb - 0.5) ds), jb = the lowest blocked station of
    the 4 (edits module header) -> (cut m, t_out s)."""
    st = LQR.simulate(src[None], torch.tensor([[float(v), 0.0]]), dtype=torch.float32)
    o = RT.executed_slack(st, geom, Bf[None], read="any4", per_state=True, tbin_s=0.25)
    fr = {k: x.float() for k, x in RT.parent_frame(geom).items()}
    if key == "nc":
        n, delta = int(o["state_nc"][0].argmin()), 0.0
    else:
        n, kd = divmod(int(o["state_ttc"][0].reshape(-1).argmin()), len(RT.TTC_DELTAS))
        delta = float(RT.TTC_DELTAS[kd])
    x = st[0, n]
    pt = x[:2] + abs(float(x[3])) * delta * torch.stack([torch.cos(x[2]), torch.sin(x[2])])
    _, _, _, jn, in_ = RT._cells(fr, pt[None, None], read="any4")
    jn, in_ = jn[0, 0], in_[0, 0]
    prof4 = Bf[jn, in_]                                                   # [4, M]
    nz = torch.nonzero(prof4.any(0))[:, 0]
    t_out = 0.0 if nz.numel() == 0 else (math.inf if int(nz[-1]) == M - 1 else (int(nz[-1]) + 1) * 0.25)
    blk4 = prof4.any(-1)
    jb = int(jn[blk4].min()) if bool(blk4.any()) else int(jn.min())
    return max(0.0, (jb - 0.5) * float(fr["ds"][0])), t_out


def test_edits_any4_vs_nearest():
    """blocked cells at d = +-0.6 m only (lateral indices 7 / 9; the straight path at d = 0 reads index 8): the nearest
    read sees a free path (no edit), the any4 read (OR over the indices 8 and 7 or 9) a blocked one -> a lon edit that
    reaches the independent any4 target (minimal up to the bisection step) and is clear under the any4 GT reduction."""
    v = 10.0
    par = TE._draft(v)
    pool = par[None, None]
    Bf = _cells_where(par, 20.0, 30.0, 0.0, 2.5, lambda d: (d.abs() - 0.6).abs() < 1e-3)
    blk, D, geom = _fields(par, [Bf, Bf])
    ce_n, po_n, _ = TE._run(_net(edit_k=1), pool, pool, TE._status(v), blk, D)
    assert ce_n["edit_kind"].tolist() == [[0]] and float(po_n["haz"]["slack_nc"][0, 0]) >= 0
    assert torch.equal(ce_n["edit_traj"][0, 0], par)
    ce, po, _ = TE._run(_net(edit_k=1, rg_read="any4"), pool, pool, TE._status(v), blk, D)
    assert float(po["haz"]["slack_nc"][0, 0]) < 0 and float(po["haz"]["slack_ttc"][0, 0]) < 0
    assert ce["edit_kind"].tolist() == [[1]] and bool(ce["edit_clear"].all()) and float(ce["d_lat"][0, 0]) == 0.0
    a = float(ce["a_lon"][0, 0])
    edit = ce["edit_traj"][0, 0]
    ex = LQR.simulate(torch.stack([par, edit]), torch.tensor([[v, 0.0]] * 2), dtype=torch.float32)
    gt = RT.executed_slack(ex, corridor_geometry(torch.stack([par, par])), Bf[None].expand(2, -1, -1, -1),
                           read="any4", tbin_s=0.25)
    assert gt["hit_nc"].tolist() == [True, False] and gt["hit_ttc"].tolist() == [True, False]
    (c_nc, to_nc), (c_tt, to_tt) = _any4_target(par, v, geom, Bf, "nc"), _any4_target(par, v, geom, Bf, "ttc")
    t_nc = TE._arrivals(edit[None], v, geom, c_nc)[0]
    t_tt = TE._arrivals(edit[None], v, geom, c_tt)[1]
    assert float(t_nc[0]) >= to_nc + 0.2 - 1e-6 and float(t_tt[0]) >= to_tt + 0.2 - 1e-6
    weaker = ED.decode_rule(par[None], torch.tensor([a + 0.016]), torch.zeros(1), torch.tensor([v]))
    w_nc, w_tt = TE._arrivals(weaker, v, geom, c_nc)[0], TE._arrivals(weaker, v, geom, c_tt)[1]
    assert float(w_nc[0]) < to_nc + 0.2 or float(w_tt[0]) < to_tt + 0.2          # minimal up to the bisection step
    # the nearest read would need no braking at all; a full-width block of the same window brakes about as hard
    full = _cells_where(par, 20.0, 30.0, 0.0, 2.5, lambda d: d.abs() <= 2.4)
    ce_f, _, _ = TE._run(_net(edit_k=1, rg_read="any4"), pool, pool, TE._status(v), *_fields(par, [full, full])[:2])
    assert ce_f["edit_kind"].tolist() == [[1]] and abs(float(ce_f["a_lon"][0, 0]) - a) < 0.1


def test_edits_b_vs_bp_channel_rule():
    """rg_stopped_aware (nearest read): NC reads B, TTC reads Bp at delta 0 / |v| delta < 2 m and U above -- a block in
    B alone is NC-limited (the rear axle's arrival), in Bp alone TTC-limited (the delta-0 read on Bp, crit_chan 2), in U
    alone TTC-limited on U (crit_chan 1): the edit follows the group the readout's rule reads, and a TTC target brakes
    harder than the NC one."""
    v = 10.0
    par = TE._draft(v)
    pool = par[None, None]
    Bf = _cells_where(par, 20.0, 30.0, 0.0, 2.5, lambda d: d.abs() <= 2.4)
    F0 = torch.zeros_like(Bf)
    res = {}
    for name, groups, chan in (("B", [Bf, F0, F0], None), ("Bp", [F0, F0, Bf], CH_BP), ("U", [F0, Bf, F0], CH_U)):
        blk, D, geom = _fields(par, groups)
        ce, po, _ = TE._run(_net(edit_k=1, rg_stopped_aware=True), pool, pool, TE._status(v), blk, D)
        h = po["haz"]
        assert ce["edit_kind"].tolist() == [[1]] and bool(ce["edit_clear"].all()), name
        assert (float(h["slack_nc"][0, 0]) < 0) == (name == "B") and (float(h["slack_ttc"][0, 0]) < 0) == (name != "B")
        if chan is not None:
            assert int(h["crit_chan"][0, 0]) == chan
        res[name] = float(ce["a_lon"][0, 0])
        cut = int(h["crit_station"][0, 0, 0 if name == "B" else 1]) * float(RT.parent_frame(geom)["ds"][0])
        t_nc, t_tt = TE._arrivals(ce["edit_traj"][0], v, geom, cut)
        assert float((t_nc if name == "B" else t_tt)[0]) >= 2.5 + 0.2 - 1e-6, name
    assert res["B"] > res["Bp"] and res["B"] > res["U"]                     # NC (B) needs less braking than TTC
    # the same Bp field read by the R1 net (2 groups: no Bp, TTC reads B = free) -> no edit
    blk, D, _ = _fields(par, [F0, F0])
    ce1, _, _ = TE._run(_net(edit_k=1), pool, pool, TE._status(v), blk, D)
    assert ce1["edit_kind"].tolist() == [[0]]
    # a pool output without the net's groups is refused (and a 2-group request on a 3-group output)
    blk3, D3, _ = _fields(par, [F0, F0, Bf])
    po = TE._run(_net(edit_k=1, rg_stopped_aware=True), pool, pool, TE._status(v), blk3, D3)[1]
    i0 = torch.zeros(1, 1, dtype=torch.long)
    assert ED._gather_sources(po, i0, pool, 1, M, 3)[0].shape[-1] == 3 * M
    with pytest.raises(ValueError, match="rg_stopped_aware"):
        ED._gather_sources(po, i0, pool, 1, M, 2)
    po["haz"].pop("bp_logit_u")
    with pytest.raises(ValueError, match="rg_stopped_aware"):
        ED._gather_sources(po, i0, pool, 1, M, 3)


@pytest.mark.parametrize("read", ["nearest", "any4"])
def test_edits_r2_with_bp_equal_b_is_r1(read):
    """with Bp = B the R2 readout's reads are the R1 ones (TTC's B read re-routed to an identical group): the edits of
    an rg_stopped_aware net on [B, U, B] equal those of the plain net on [B, U] bitwise, per source (B and U differ;
    3 parents, all shortlisted), for both reads."""
    v = 10.0
    pool = torch.stack([TE._draft(v), TE._draft(9.0, 0.004), TE._draft(11.0, -0.004)])[None]
    par = pool[0, 0]
    Bf = _cells_where(par, 20.0, 30.0, 0.0, 2.5, lambda d: d.abs() <= 1.2)
    Uf = _cells_where(par, 16.0, 40.0, 0.0, 3.0, lambda d: (d >= -0.6) & (d <= 3.0))
    _, Dc, _ = TE._fields(par, curb=(4.0, TE._narrow(0.9)))
    out = {}
    for sa, groups in ((False, [Bf, Uf]), (True, [Bf, Uf, Bf])):
        blk, _, _ = _fields(par, groups)
        ce, _, _ = TE._run(_net(edit_k=3, rg_read=read, rg_stopped_aware=sa), pool, pool, TE._status(v), blk, Dc)
        out[sa] = _sorted(ce)
    assert int((out[False]["edit_kind"] > 0).sum()) >= 2
    for k in out[False]:
        assert torch.equal(out[False][k], out[True][k]), k


def test_any4_targets_match_hard_evidence_and_independent_window():
    """any4_targets on random 3-group fields and random executed states: the critical read's nearest station =
    hard_evidence(read='any4') crit_station on every hit row; the OR window and the cut = an explicit per-row loop."""
    g = torch.Generator().manual_seed(7)
    n = 24
    pars = torch.stack([TE._draft(6.0 + 0.3 * i, 0.01 * (i % 5 - 2)) for i in range(n)])
    fr = {k: x.float() for k, x in RT.parent_frame(corridor_geometry(pars)).items()}
    blk_b = torch.rand(n, 48, 17, 3 * M, generator=g) < 0.06
    D = torch.full((n, 48, 17), 3.0)
    st = LQR.simulate(pars, torch.stack([torch.full((n,), 7.0), torch.zeros(n)], -1), dtype=torch.float32)
    tq = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    rows = torch.arange(n)
    geo = read_geometry(fr, st, tq, M, 0.25, ttc_b_chan=CH_BP)
    he = hard_evidence(blk_b, D, rows, geo, M, 0.25, read="any4")
    a4 = ED.any4_targets(blk_b, rows, geo, M, 0.25, fr["ds"])
    hits = 0
    for key, col, sl in (("nc", 0, he["slack_nc"]), ("ttc", 1, he["slack_ttc"])):
        for r in torch.nonzero(sl < 0)[:, 0].tolist():
            hits += 1
            assert int(a4[f"a4_station_{key}"][r]) == int(he["crit_station"][r, col]), (key, r)
            # explicit: the arg-min point of the read, its 4 cells, the group of its channel
            gk = geo[key]
            from navsim.agents.para_ssr.ck.v4.range_readout import any4_cells, hard_slack
            s_, _, _ = hard_slack(blk_b, rows, gk, geo[f"t_{key}"], 0 if key == "nc" else geo["chan"],
                                  geo[f"use_{key}"], M, 0.25, read="any4")
            q = int(s_[r].argmin())
            jn, in_, _ = any4_cells(gk)
            grp = 0 if key == "nc" else int(geo["chan"][r, q])
            prof = torch.stack([blk_b[r, int(jn[r, q, c]), int(in_[r, q, c]), grp * M:(grp + 1) * M] for c in range(4)])
            nz = torch.nonzero(prof.any(0))[:, 0]
            t_out = 0.0 if nz.numel() == 0 else (math.inf if int(nz[-1]) == M - 1 else (int(nz[-1]) + 1) * 0.25)
            assert float(a4[f"a4_tout_{key}"][r]) == t_out, (key, r)
            jb = min(int(jn[r, q, c]) for c in range(4) if bool(prof[c].any()))
            assert abs(float(a4[f"a4_cut_{key}"][r]) - max(0.0, (jb - 0.5) * float(fr["ds"][r]))) < 1e-5, (key, r)
    assert hits >= 10


# ----------------------------------------------------------------------------------------------- (c) online2
def test_field_groups_and_occ_channels_config():
    assert O.OCC_CHANNELS == ("B", "BU", "BBp", "all")
    assert O.field_groups(True) == ("B", "U") and O.field_groups("BBp") == ("B", "Bp")
    assert O.field_groups("all") == ("B", "U", "Bp") and O.field_groups(["Bp", "B"]) == ("Bp", "B")
    for bad in ("Q", ["B", "B"], ["X"], []):
        with pytest.raises(ValueError):
            O.field_groups(bad)
    assert O.kd_constraint_request("B") is True and O.kd_constraint_request("BU") is True
    assert O.kd_constraint_request("BBp") == "BBp" and O.kd_constraint_request("all") == "all"
    assert O.net_field_groups(V4Cfg.range_arm(**R2)) == ("B", "U", "Bp")
    assert O.net_field_groups(V4Cfg.range_arm()) == ("B", "U")
    # occ_channels with the Bp group: R2 students only
    base = dict(TL.cfg2(Path("/tmp/x")).to_dict(), io_dir="/tmp/x")
    for occ in ("BBp", "all"):
        r1 = dict(base, v4=dict(base["v4"], **TW.V43, kd_constraint={"enabled": True, "occ_channels": occ}))
        with pytest.raises(ValueError, match="R2 only"):
            O.CKE2E2Config.from_any(r1).validate(None)
        r2 = dict(base, v4=dict(base["v4"], **V43_R2, kd_constraint={"enabled": True, "occ_channels": occ}))
        O.CKE2E2Config.from_any(r2).validate(None)
    O.CKE2E2Config.from_any(dict(base, v4=dict(base["v4"], **V43_R2, kd_constraint={"enabled": True}))).validate(None)


def test_constraint_kd_groups():
    """'BBp' / 'all': the per-group BCE-KL of the student's B / U / Bp against the teacher's groups of the same name
    (any teacher group order), mean over the DET-ok rows; a missing group on either side is refused."""
    g = torch.Generator().manual_seed(2)
    B, K, U, Ut = 2, 5, 3, 4
    haz = {k: torch.randn(B, U, 48, 17, M, generator=g, requires_grad=True)
           for k in ("occ_logit_u", "ovl_logit_u", "bp_logit_u")}
    haz.update(margin_u=torch.randn(B, U, 48, 17, generator=g, requires_grad=True),
               parent_index=torch.tensor([[0, 0, 1, 2, 1], [2, 1, 1, 0, 0]]))
    pt = torch.tensor([[3, 3, 0, 1, 0], [1, 2, 2, 0, 0]])
    s_u = torch.arange(48, dtype=torch.float32)[None, None].expand(B, Ut, 48) + 0.5
    tf = {g_: 3 * torch.randn(B, Ut, 48, 17, M, generator=g) for g_ in ("B", "U", "Bp")}
    fm = {"blk_u": torch.zeros(B, Ut, 48, 17, M), "groups": ("B",), "margin_u": torch.zeros(B, Ut, 48, 17),
          "parent_index": pt, "s_u": s_u}
    ok = torch.ones(B, K, dtype=torch.bool)
    pairs = [(0, 0, 3), (0, 1, 0), (0, 2, 1)]                                # DET ok on token 0 only
    key = O.FIELD_GROUP_KEY
    for occ in ("BBp", "all"):
        want = O.CN_GROUPS[occ]
        for tg in (want, ("Bp", "U", "B")):                                  # the teacher's own group order
            fd = {"blk_u": torch.cat([tf[x] for x in tg], -1), "groups": tg, "margin_u": torch.zeros(B, Ut, 48, 17),
                  "parent_index": pt, "s_u": s_u}
            T = {"fields_det": fd, "fields_map": fm, "ok_det_tok": torch.tensor([True, False]),
                 "ok_map_tok": torch.tensor([True, True])}
            l_o, _, st = O.constraint_kd(haz, T, ok, {"occ_channels": occ, "map_range_m": 32.0})
            e = torch.stack([TW._bce_kl(torch.cat([haz[key[x]][b, u] for x in want], -1),
                                        torch.cat([tf[x][b, ut] for x in want], -1)).mean()
                             for b, u, ut in pairs]).mean()
            assert torch.allclose(l_o, e, atol=1e-6) and st["cn_det_rows"] == 3, (occ, tg)
    # 'B' with a B-only teacher dict = 'B' with the B | U dict (the request does not change the term)
    fdb = {"blk_u": tf["B"], "groups": ("B",), "margin_u": torch.zeros(B, Ut, 48, 17), "parent_index": pt, "s_u": s_u}
    fdbu = dict(fdb, blk_u=torch.cat([tf["B"], tf["U"]], -1), groups=("B", "U"))
    T = {"fields_map": fm, "ok_det_tok": torch.tensor([True, False]), "ok_map_tok": torch.tensor([True, True])}
    lb = O.constraint_kd(haz, dict(T, fields_det=fdb), ok, {"occ_channels": "B", "map_range_m": 32.0})[0]
    lbu = O.constraint_kd(haz, dict(T, fields_det=fdbu), ok, {"occ_channels": "B", "map_range_m": 32.0})[0]
    assert torch.allclose(lb, lbu, atol=1e-7)
    with pytest.raises(ValueError, match="teacher Bp"):
        O.constraint_kd(haz, dict(T, fields_det=fdbu), ok, {"occ_channels": "BBp", "map_range_m": 32.0})
    h1 = {k: v for k, v in haz.items() if k != "bp_logit_u"}
    fdall = dict(fdb, blk_u=torch.cat([tf[x] for x in ("B", "U", "Bp")], -1), groups=("B", "U", "Bp"))
    with pytest.raises(ValueError, match="student Bp"):
        O.constraint_kd(h1, dict(T, fields_det=fdall), ok, {"occ_channels": "all", "map_range_m": 32.0})


@pytest.fixture(scope="module")
def r2_teachers(tmp_path_factory):
    if not TL.have_inputs():
        pytest.skip("CK2 smoke teachers absent")
    root = tmp_path_factory.mktemp("r2_teachers")
    r2 = V4Cfg.range_arm(**R2)
    return {"T": TW._range_run(root, "T", v4=r2, name="T_r2"), "M": TW._range_run(root, "M", v4=r2, name="M_r2"),
            "T_r1": TW._range_run(root, "T", name="T_r1"), "M_r1": TW._range_run(root, "M", name="M_r1"),
            "T_any4": TW._range_run(root, "T", v4=V4Cfg.range_arm(rg_read="any4"), name="T_any4"), "root": root}


@have
def test_teacher_field_groups(r2_teachers):
    t = r2_teachers
    tp = O.TeacherPair2(str(t["T"]), str(t["M"]), require_done=True, amp=False, fields=True)
    f, tg = TL.fresh(2)
    top = TL.fake_predictions(2, seed=1)
    cand = top["trajectory_anchors"][None, :8].expand(2, 8, 8, 3).float() + 0.01
    from navsim.agents.para_ssr.ck.cands2 import online_variants
    V = online_variants(cand.double(), torch.tensor([5.0, 5.0], dtype=torch.float64), None)
    pool, par = V["traj96"][:, :24].float(), cand.repeat_interleave(6, 1)[:, :24]
    args = (tg["kd_bev_0"], tg["kd_ok_0"], tg["kd_bev_1"], tg["kd_ok_1"], pool, f["status_feature"].float())
    r = {spec: tp.run(*args, parents=par, fields=spec) for spec in (True, "B", "BBp", "all")}
    for spec, want in ((True, ("B", "U")), ("B", ("B",)), ("BBp", ("B", "Bp")), ("all", ("B", "U", "Bp"))):
        fd = r[spec]["fields_det"]
        assert fd["groups"] == want and fd["blk_u"].shape[-1] == len(want) * M, spec
        assert r[spec]["fields_map"]["groups"] == want
        assert torch.equal(r[spec]["kd_score_prob"], r[True]["kd_score_prob"])        # scores unchanged
    a = r["all"]["fields_det"]["blk_u"]
    assert torch.equal(r[True]["fields_det"]["blk_u"], a[..., :2 * M]) and torch.equal(r["B"]["fields_det"]["blk_u"],
                                                                                        a[..., :M])
    assert torch.equal(r["BBp"]["fields_det"]["blk_u"], torch.cat([a[..., :M], a[..., 2 * M:]], -1))
    assert not torch.equal(a[..., :M], a[..., 2 * M:])                                   # Bp is its own group
    # Bp = the teacher's own bp_logit_u (its score_set)
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    ego = ego_inputs(f["status_feature"].float())
    with torch.no_grad():
        _, _, fl = O.teacher_forward(tp.det, tg["kd_bev_0"].float(), pool, ego, parents=par, return_fields=["Bp"])
        feat, mem = tp.det.trunk.scene(tg["kd_bev_0"].float(), 2)
        o = tp.det.score_set(feat, mem, ego, pool, par, pref=tp.det.scene_pref(mem, ego), return_pctx=True)
    assert torch.equal(fl["blk_u"], o["haz"]["bp_logit_u"].float()) and torch.equal(fl["blk_u"], a[..., 2 * M:])
    # an R1 teacher has no Bp group
    tp1 = O.TeacherPair2(str(t["T_r1"]), str(t["M_r1"]), require_done=True, amp=False, fields=True)
    with pytest.raises(ValueError, match="Bp"):
        tp1.run(*args, parents=par, fields="BBp")


def _r2_cfg(tmp, t, v4=None, **kw):
    return TL.cfg2(tmp, v4=dict(V43_R2 if v4 is None else v4, kd_constraint={"enabled": True, **kw.pop("kc", {})}),
                   kd_score_keys=["nc", "dac"], lat_kd={"start_mb": 0}, teacher_det_run=str(t["T"]),
                   teacher_map_run=str(t["M"]), **kw)


@have
def test_runtime_check_r2_pairs(r2_teachers, tmp_path):
    """check_constraint_teachers: an R2 student with R2 teachers passes; so do a student whose rg_read differs from
    the teachers' with the same B rule (R2 teachers + a nearest-read stopped-aware student, an any4 plain-B teacher + an
    R1 student: the recorded decision, rg_read is not a field property); an R1 student with R2 teachers, the R2
    student with R1 teachers, a plain-B any4 teacher for a stopped-aware student and a mixed pair are refused, naming
    the mismatching key and teacher and never rg_read."""
    t = r2_teachers
    O.CKE2E2(_r2_cfg(tmp_path / "ok", t)).teachers("cpu")
    O.CKE2E2(_r2_cfg(tmp_path / "ok_bbp", t, kc={"occ_channels": "BBp"})).teachers("cpu")
    O.CKE2E2(_r2_cfg(tmp_path / "ok_read", t, v4=dict(V43_R2, rg_read="nearest"))).teachers("cpu")
    O.CKE2E2(_r2_cfg(tmp_path / "ok_read1", dict(t, T=t["T_any4"], M=t["M_r1"]), v4=TW.V43)).teachers("cpu")
    cases = [("r1_student", _r2_cfg(tmp_path / "a", t, v4=TW.V43), ("det", "map", "rg_stopped_aware")),
             ("r1_teachers", _r2_cfg(tmp_path / "b", dict(t, T=t["T_r1"], M=t["M_r1"])),
              ("det", "map", "rg_stopped_aware")),
             ("plain_b_any4", _r2_cfg(tmp_path / "c", dict(t, T=t["T_any4"]), v4=dict(TW.V43, rg_stopped_aware=True)),
              ("det", "rg_stopped_aware")),
             ("mixed", _r2_cfg(tmp_path / "d", dict(t, M=t["M_r1"])), ("map", "rg_stopped_aware"))]
    for name, c, words in cases:
        with pytest.raises(ValueError) as ei:
            O.CKE2E2(c).teachers("cpu")
        msg = str(ei.value)
        assert msg.startswith(O.CONSTRAINT_REFUSAL) and all(w in msg for w in words), (name, msg)
        assert "rg_read" not in msg, (name, msg)
    assert "det" not in str(ei.value).split(O.CONSTRAINT_REFUSAL, 1)[1]          # the mixed pair: only the MAP named


@have
def test_r2_student_loss_steps_warmup_and_onpolicy(r2_teachers, tmp_path):
    """the v4_3 R2 student with the R2 teachers: one warm-up loss step (constraint KD on, its only difference to the
    same step with it off, gradients into the B / U / Bp head), and one on-policy step whose computed edits (R2 reads)
    are recorded with the 96 (labeler2's input)."""
    t = r2_teachers
    out = {}
    for on in (False, True):
        c = TL.cfg2(tmp_path / f"on{int(on)}", v4=dict(V43_R2, kd_constraint={"enabled": on}),
                    kd_score_keys=["nc", "dac"], lat_kd={"start_mb": 0}, teacher_det_run=str(t["T"]),
                    teacher_map_run=str(t["M"]))
        c.validate(None)
        ck = O.CKE2E2(c)
        st = TLOSS._student(c)
        assert st.v4 == V4Cfg.range_arm(edits="computed", **R2)
        f, tt = TL.fresh(2)
        loss, logs = ck.loss(st, f, tt, TL.fake_predictions(2, seed=0))
        loss.backward()
        out[on] = (float(loss), {k: float(v) for k, v in logs.items()}, st)
    (l0, g0, _), (l1, g1, st1) = out[False], out[True]
    assert math.isfinite(l0) and math.isfinite(l1) and g1["v4/cn_det_rows"] > 0 and g1["v4/cn_map_rows"] > 0
    assert abs(g1["ck2/bce"] - g0["ck2/bce"]) < 1e-6 and abs(g1["ck2/kd_score"] - g0["ck2/kd_score"]) < 1e-6
    assert abs((l1 - l0) - g1["ck2/wt_kd_constraint"]) < 1e-4
    w = st1.v4_readout.blk_head[2].weight.grad
    assert float(w[:M].abs().sum()) > 0                                       # B rows (distilled, occ_channels B)
    for p in st1.parameters():
        assert p.grad is None or bool(torch.isfinite(p.grad).all())
    # on-policy step with the computed edits recorded (R2 reads in ck/v4/edits.py)
    c, ck, _, _ = TLOSS._onpolicy(tmp_path / "op", lat_kd={"start_mb": 0}, rec_chunk_tokens=1,
                                  v4=dict(V43_R2, kd_constraint={"enabled": True}), kd_score_keys=["nc", "dac"],
                                  teacher_det_run=str(t["T"]), teacher_map_run=str(t["M"]))
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target, with_edits=True, with_pref=True)   # as the config's own store
    assert O.CKE2E2(c).labels().with_edits
    st = TLOSS._student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 5, c.rec_chunk_tokens)
    f, tt = TL.fresh(2, rows=[3, 7])
    loss, logs = ck.loss(st, f, tt, TL.fake_predictions(2, seed=3))
    loss.backward()
    assert torch.isfinite(loss) and float(logs["ck2/phase"]) == 2.0 and "v4/edit_rec_valid_frac" in logs
    assert 0.0 <= float(logs["v4/edit_clear_frac"]) <= 1.0
    assert "v4/cn_occ" in logs and np.isfinite(float(logs["v4/cn_occ"]))
    ck.recorder.close_epoch()
    recs = {}
    for ch in ck.recorder.chunks:
        d = D2.read_rec2_chunk(Path(c.io_dir) / "rec" / "ep005" / "r0" / ch)
        for i, r in enumerate(d["row"]):
            recs[int(r)] = {k: d[k][i] for k in ("traj", "edit_traj", "edit_src", "edit_valid")}
    assert set(recs) == {3, 7} and not recs[3]["edit_valid"].any()          # v0 < 2 m/s: no edit
    r7 = recs[7]
    assert float(logs["v4/edit_rec_lon_frac"]) > 0 and r7["edit_valid"].any()   # edits of the moving token recorded
    assert bool((r7["edit_src"] % 6 == 0).all())                             # sources = identity columns
    for j in np.flatnonzero(r7["edit_valid"]):                               # a recorded edit differs from its source
        assert not np.array_equal(r7["edit_traj"][j], r7["traj"][int(r7["edit_src"][j])])
    assert abs(float(logs["v4/edit_rec_valid_frac"]) - float(np.mean([recs[r]["edit_valid"].mean() for r in recs]))) \
        < 1e-6


# ----------------------------------------------------------------------------------------------- (d) + (e) launch
def _arm_sets(tmp_path, t, extra=(), name="ck2e2e_v43"):
    """the v4_3 arm file + a user set file (teachers t, kd_calib off, extra lines) -> set lines."""
    from tools.ck.e2e2 import launch_util2 as L2
    return L2.LU.read_lines(str(L2.V4_3_SET)) + L2.LU.read_lines(str(TW._set_file(tmp_path / f"{name}.set", t, extra)))


def _launch(tmp_path, t, extra=(), name="ck2e2e_v43"):
    from tools.ck.e2e2 import launch_util2 as L2
    return L2.build_overrides2(str(tmp_path / name), ["0", "1"], ck_sets=_arm_sets(tmp_path, t, extra, name),
                               wandb="disable")


@have
def test_v4_3_arm_is_r2_and_launch_checks(r2_teachers, tmp_path):
    """the arm file: R2 student, teacher paths = the R2 run names (refused until they exist; calibration placeholders);
    with R2 teachers the launch passes and records both sides; an R1 student (set lines) with R2 teachers and the R2
    arm with R1 teachers are refused before launch with the runtime message."""
    from tools.ck.e2e2 import launch_util2 as L2
    t = r2_teachers
    lines = L2.LU.read_lines(str(L2.V4_3_SET))
    kv = dict(x.split("=", 1) for x in lines)
    assert kv["v4.rg_read"] == "any4" and kv["v4.rg_stopped_aware"] == "true"
    assert kv["v4.kd_constraint.occ_channels"] == "B"
    assert kv["teacher_det_run"].endswith("/v4/train/v4T10_rng2")
    assert kv["teacher_map_run"].endswith("/v4/train/v4M10_rng2")
    assert L2.PLACEHOLDER in kv["kd_calib.det"] and L2.PLACEHOLDER in kv["kd_calib.map"]
    assert "BBp" in L2.V4_3_SET.read_text()                                  # the option is named in the comment
    with pytest.raises(L2.LaunchError, match="PLACEHOLDER"):                 # calibration files not given
        L2.build_overrides2(str(tmp_path / "ph"), ["0", "1"], ck_sets=lines, wandb="disable")
    rd = Path(kv["teacher_det_run"])
    if not (rd / "done.json").is_file():                                     # the R2 runs do not exist yet: refused
        with pytest.raises(L2.LaunchError):
            L2.build_overrides2(str(tmp_path / "nr"), ["0", "1"], ck_sets=lines + ["kd_calib.enabled=false",
                                                                                   "teacher_ep_check=false"],
                                wandb="disable")
    ov, ck, tj = _launch(tmp_path, t)
    c = O.CKE2E2Config.from_any(ck)
    assert c.v4_net() == V4Cfg.range_arm(edits="computed", **R2)
    kq = tj["kd_constraint"]
    assert kq["student_r2"] == {"rg_read": "any4", "rg_stopped_aware": True}
    assert all(kq[n]["rg_read"] == "any4" and kq[n]["rg_stopped_aware"] for n in ("det", "map"))
    assert kq["rg_read_differs"] == []
    assert any(o.endswith("ck_e2e2.v4.rg_read=any4") for o in ov)
    # refused at launch with the runtime message
    for name, tt, extra in (("r1_student", t, ["v4.rg_read=nearest", "v4.rg_stopped_aware=false"]),
                            ("r1_teachers", dict(t, T=t["T_r1"], M=t["M_r1"]), [])):
        with pytest.raises(L2.LaunchError) as ei:
            _launch(tmp_path, tt, extra, name=name)
        msg = str(ei.value)
        assert msg.startswith(O.CONSTRAINT_REFUSAL) and "rg_stopped_aware" in msg and "rg_read" not in msg, msg
        eff = L2.apply_sets2(L2.load_ck2_yaml(), _arm_sets(tmp_path, tt, extra, name))
        eff["io_dir"] = str(tmp_path / name / "io")
        with pytest.raises(ValueError) as er:
            O.CKE2E2(L2.ck2_from_any(eff)).teachers("cpu")
        assert str(er.value) == msg                                          # the same text at runtime
    # rg_read is not part of the rule: the R2 teachers with a nearest-read stopped-aware student launch, the record
    # names both teachers in rg_read_differs, and the runtime accepts the same effective config
    extra = ["v4.rg_read=nearest"]
    ov, ck, tj = _launch(tmp_path, t, extra, name="read_nearest")
    kq = tj["kd_constraint"]
    assert kq["student_r2"] == {"rg_read": "nearest", "rg_stopped_aware": True}
    assert kq["rg_read_differs"] == ["det", "map"]
    eff = L2.apply_sets2(L2.load_ck2_yaml(), _arm_sets(tmp_path, t, extra, "read_nearest"))
    eff["io_dir"] = str(tmp_path / "read_nearest" / "io")
    O.CKE2E2(L2.ck2_from_any(eff)).teachers("cpu")
