"""refiner v4_3 wiring (docs/refiner_v4/V4_3_SPEC.md M5 / M6 integration): the ck_e2e2 knobs (the OPTIONAL v4-block
keys kd_score_scope, kd_constraint and the V4Cfg v4_3 keys, absent by default: older configs unchanged), the constraint
KD (masks / values / teacher fields of range teachers, refusal of other teachers), kd_score_scope 'near' (lead ttc0 and
parent corner SDF from the ref_* GT; missing ref_* keys refused), the computed edits of a range student recorded ->
labelled by labeler2 -> in the next epoch's label set (and the infer outputs + infer_check; no-edit slots not
selectable), the 96 + K evaluation with the val-frozen beta on a synthetic eval root (no-edit copies never selected,
another navtest K refused), infer_select.edits (96 + K in the agent), the v4_3 arm (launch dry runs, placeholders
refused), run_teachers.sh READOUT=range, train_ck2 --v4-readout range (3 CPU steps on real data with --time-idx-root,
both arms) and the navtest r34 teacher eval on a range teacher.  CPU only:

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_wiring.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
import test_ck2_e2e_labeler as TLAB  # noqa: E402
import test_ck2_e2e_loss as TLOSS  # noqa: E402
import test_ck2_e2e_select as TSEL  # noqa: E402
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.model import CKNet, load_arm_norm, save_ck  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import V4Cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CK = Path(Cn.CK_DATA)
TIDX = CK / "v4" / "time_idx"
have = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
V43 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True, "pref_head": True,
       "corridor_version": 2, "edits": "computed", "kd_parent_only": True}
M = 20


def _range_run(root: Path, arm: str, cv: int = 2, v4: V4Cfg = None, name: str = None) -> Path:
    """a finished (done.json) 'train_ck2' range teacher run of arm T / M with random weights (lon head zero), the smoke
    teachers' z-score files."""
    src = TL.SMOKE_T if arm == "T" else TL.SMOKE_M
    rd = Path(root) / (name or f"rng{arm}_cv{cv}")
    if (rd / "done.json").is_file():
        return rd
    rd.mkdir(parents=True, exist_ok=True)
    nf = "norm.npz" if arm == "T" else "norm_map.npz"
    shutil.copy(src / nf, rd / nf)
    v4 = v4 or V4Cfg.range_arm()
    torch.manual_seed(5)
    net = CKNet(arm, seed=1 if arm == "T" else 2, norm=load_arm_norm(rd, arm), v4=v4, corridor_version=cv)
    with torch.no_grad():
        net.trunk.lon_head[-1].weight.zero_()
        net.trunk.lon_head[-1].bias.zero_()
    cfg = {"arm": arm, "seed": 1 if arm == "T" else 2, "trainer": "train_ck2", "v4": v4.to_dict(), "corridor_version": cv,
           "score_hidden": 256, "lead_aux": 0}
    save_ck(rd / "ckpt_last.pt", net, cfg)
    (rd / "config.json").write_text(json.dumps(cfg))
    (rd / "done.json").write_text("{}")
    return rd


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    if not TL.have_inputs():
        pytest.skip("CK2 smoke teachers absent")
    root = tmp_path_factory.mktemp("range_teachers")
    r2 = V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True)          # the Phase 0b R2 teachers (v4_3 arm)
    return {"T": _range_run(root, "T"), "M": _range_run(root, "M"), "T_cv1": _range_run(root, "T", cv=1),
            "T_r2": _range_run(root, "T", v4=r2, name="rngT_r2"), "M_r2": _range_run(root, "M", v4=r2, name="rngM_r2"),
            "root": root}


# ----------------------------------------------------------------------------------------------- config
def test_config_defaults_off_and_v43_keys():
    c = O.CKE2E2Config()
    assert c.kd_score_scope == "all" and c.kd_constraint["enabled"] is False
    assert set(c.v4) == set(O._d_v4()) and not set(c.v4) & set(O._d_v4_opt())     # optional keys absent by default
    assert c.v4_net() == V4Cfg() and not c.v4_net().on
    assert O.CKE2E2Config.from_any(c.to_dict()) == c
    # the v4_3 student network = the spec range arm + computed edits; the optional keys round trip
    d = dict(c.to_dict(), enabled=False, v4=dict(c.v4, **V43, kd_constraint={"enabled": True}),
             kd_score_keys=["nc", "dac"])
    c3 = O.CKE2E2Config.from_any(d)
    c3.validate(None)
    assert c3.v4_net() == V4Cfg.range_arm(edits="computed") and c3.kd_score_keys == ["nc", "dac"]
    assert c3.kd_constraint == dict(O._d_kd_constraint(), enabled=True) and O.CKE2E2Config.from_any(c3.to_dict()) == c3
    # YAML 1.1 reads an unquoted off as False (the launcher's set-file parser): taken as 'off'
    assert O.CKE2E2Config.from_any({"v4": {"vmax": False, "edits": False}}).v4["vmax"] == "off"
    with pytest.raises(ValueError, match="unknown"):
        O.CKE2E2Config.from_any({"v4": {"kd_constraint": {"enable": True}}})
    dv = d["v4"]
    bad = [{"v4": dict(c.v4, kd_score_scope="far")}, {"v4": dict(c.v4, kd_constraint={"enabled": True})},  # mean
           dict(d, v4=dict(dv, kd_constraint={"enabled": True, "occ_channels": "Q"})), dict(d, v4=dict(dv, edit_k=4)),
           dict(d, v4=dict(dv, kd_constraint={"enabled": True, "map_range_m": 0.0}))]
    for b in bad:
        x = {**c.to_dict(), **b, "enabled": True, "io_dir": "/tmp/x"}
        with pytest.raises(ValueError):
            O.CKE2E2Config.from_any(x).validate(None)


def test_split_haz_keeps_parent_rows():
    haz = {"occ_logit_u": torch.zeros(2, 3, 4), "margin_u": torch.zeros(2, 3), "ovl_logit_u": torch.zeros(2, 3, 4),
           "parent_index": torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 0]]), "lp_nc": torch.arange(10.).reshape(2, 5)}
    a, b = O.split_haz(haz, 0, 2), O.split_haz(haz, 2)
    assert a["occ_logit_u"] is haz["occ_logit_u"] and b["margin_u"] is haz["margin_u"]
    assert a["parent_index"].tolist() == [[0, 0], [0, 1]] and b["lp_nc"].shape == (2, 3)


# ----------------------------------------------------------------------------------------------- constraint KD
def _bce_kl(s, t):
    p = torch.sigmoid(t)
    return F.binary_cross_entropy_with_logits(s, p, reduction="none") + \
        (p * torch.log(p.clamp(min=1e-12)) + (1 - p) * torch.log((1 - p).clamp(min=1e-12)))


def test_constraint_kd_masks_and_values():
    g = torch.Generator().manual_seed(0)
    B, K, U, Ut = 2, 5, 3, 4
    haz = {"occ_logit_u": torch.randn(B, U, 48, 17, M, generator=g, requires_grad=True),
           "ovl_logit_u": torch.randn(B, U, 48, 17, M, generator=g, requires_grad=True),
           "margin_u": torch.randn(B, U, 48, 17, generator=g, requires_grad=True),
           "parent_index": torch.tensor([[0, 0, 1, 2, 1], [2, 1, 1, 0, 0]])}
    pt = torch.tensor([[3, 3, 0, 1, 0], [1, 2, 2, 0, 0]])                    # the teacher's own row order
    s_u = torch.arange(48, dtype=torch.float32)[None, None].expand(B, Ut, 48) * 1.0 + 0.5      # ds 1 m: s_j = j + 0.5
    fd = {"blk_u": 3 * torch.randn(B, Ut, 48, 17, 2 * M, generator=g), "margin_u": torch.zeros(B, Ut, 48, 17),
          "parent_index": pt, "s_u": s_u}
    fm = {"blk_u": torch.zeros(B, Ut, 48, 17, 2 * M), "margin_u": 9 * torch.randn(B, Ut, 48, 17, generator=g),
          "parent_index": pt, "s_u": s_u}
    T = {"fields_det": fd, "fields_map": fm, "ok_det_tok": torch.tensor([True, False]),
         "ok_map_tok": torch.tensor([True, True])}
    ok = torch.ones(B, K, dtype=torch.bool)
    ok[1, 3:] = False                                                        # token 1, student row 0 never counted
    kc = {"occ_channels": "BU", "map_range_m": 32.0}
    l_o, l_m, st = O.constraint_kd(haz, T, ok, kc)
    # expected: DET only token 0 (rows 0, 1, 2 <- teacher rows 3, 0, 1); MAP token 0 rows 0-2, token 1 rows 2, 1
    pairs_d = [(0, 0, 3), (0, 1, 0), (0, 2, 1)]
    pairs_m = pairs_d + [(1, 2, 1), (1, 1, 2)]
    s = lambda b, u: torch.cat([haz["occ_logit_u"][b, u], haz["ovl_logit_u"][b, u]], -1)
    e_o = torch.stack([_bce_kl(s(b, u), fd["blk_u"][b, ut]).mean() for b, u, ut in pairs_d]).mean()
    near = (s_u[0, 0] <= 32.0)[:, None].expand(48, 17)
    hub = [F.huber_loss(haz["margin_u"][b, u], fm["margin_u"][b, ut].clamp(-5, 5), delta=1.0, reduction="none")[near]
           for b, u, ut in pairs_m]
    e_m = torch.cat(hub).mean()
    assert st["cn_det_rows"] == 3 and st["cn_map_rows"] == 5
    assert torch.allclose(l_o, e_o, atol=1e-5) and torch.allclose(l_m, e_m, atol=1e-5)
    # occ_channels 'B': only the not-behind channel
    l_b, _, _ = O.constraint_kd(haz, T, ok, dict(kc, occ_channels="B"))
    e_b = torch.stack([_bce_kl(haz["occ_logit_u"][b, u], fd["blk_u"][b, ut][..., :M]).mean()
                       for b, u, ut in pairs_d]).mean()
    assert torch.allclose(l_b, e_b, atol=1e-5)
    # a student equal to its teacher: zero KL, zero Huber (inside the clip)
    h2 = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in haz.items()}
    for b, u, ut in pairs_m:
        h2["occ_logit_u"][b, u] = fd["blk_u"][b, ut][..., :M]
        h2["ovl_logit_u"][b, u] = fd["blk_u"][b, ut][..., M:]
        h2["margin_u"][b, u] = fm["margin_u"][b, ut].clamp(-5, 5)
    z_o, z_m, _ = O.constraint_kd(h2, T, ok, kc)
    assert abs(float(z_o)) < 1e-5 and abs(float(z_m)) < 1e-6
    (l_o + l_m).backward()
    assert float(haz["occ_logit_u"].grad[1].abs().sum()) == 0.0                # DET not ok on token 1
    assert float(haz["margin_u"].grad[1, 0].abs().sum()) == 0.0                # row 0 of token 1 not counted
    assert float(haz["margin_u"].grad[0, 0, 40].abs().sum()) == 0.0            # s = 40.5 m > 32 m
    # nothing counted: graph-connected zeros
    z1, z2, _ = O.constraint_kd(haz, T, torch.zeros(B, K, dtype=torch.bool), kc)
    assert float(z1) == 0.0 and float(z2) == 0.0 and z1.requires_grad


@have
def test_teacher_fields_and_refusals(teachers):
    tp = O.TeacherPair2(str(teachers["T"]), str(teachers["M"]), require_done=True, amp=False, fields=True)
    with pytest.raises(ValueError, match="range"):
        O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), require_done=True, amp=False, fields=True)
    plain = O.TeacherPair2(str(teachers["T"]), str(teachers["M"]), require_done=True, amp=False)
    f, t = TL.fresh(2)
    from navsim.agents.para_ssr.ck.cands2 import online_variants
    top = TL.fake_predictions(2, seed=1)
    cand = top["trajectory_anchors"][None, :16].expand(2, 16, 8, 3).float() + 0.01
    V = online_variants(cand.double(), torch.tensor([5.0, 5.0], dtype=torch.float64), None)
    pool = V["traj96"][:, :48].float()
    par = cand.repeat_interleave(6, 1)[:, :48]
    args = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], pool, f["status_feature"].float())
    r = tp.run(*args, parents=par, fields=True)
    r0 = plain.run(*args, parents=par)
    assert torch.equal(r["kd_score_prob"], r0["kd_score_prob"])                 # extra outputs: scores unchanged
    with pytest.raises(ValueError, match="fields=True"):
        plain.run(*args, parents=par, fields=True)
    fd = r["fields_det"]
    U = fd["blk_u"].shape[1]
    assert U == 8 and fd["blk_u"].shape == (2, U, 48, 17, 2 * M) and fd["margin_u"].shape == (2, U, 48, 17)
    assert fd["parent_index"].shape == (2, 48) and fd["s_u"].shape == (2, U, 48)
    assert bool((fd["parent_index"][:, ::6] == torch.arange(8)).all())          # one row per distinct parent
    assert bool(r["ok_det_tok"].all()) and bool(r["ok_map_tok"].all())
    # the station arc lengths of a row are its parent's grid (corridor v2 of the teacher)
    from navsim.agents.para_ssr.refiner.corridor import corridor_geometry
    gs = corridor_geometry(cand[0, :1], version=2).s[0]
    assert torch.allclose(fd["s_u"][0, 0], gs.float(), atol=1e-5)
    # the grid must be the student's (corridor version / time bins)
    c = TL.cfg2(Path(teachers["root"]) / "io", v4=dict(V43, kd_constraint={"enabled": True}),
                teacher_det_run=str(teachers["T_cv1"]), teacher_map_run=str(teachers["M"]))
    with pytest.raises(ValueError, match="grid"):
        O.CKE2E2(c).teachers("cpu")
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    hz = CKNet("S", seed=0, v4=V4Cfg(readout="hazard"))
    with pytest.raises(ValueError, match="range"):
        O.teacher_forward(hz, torch.randn(1, 256, 50, 100), pool[:1], ego_inputs(f["status_feature"][:1].float()),
                          return_fields=True)


@have
def test_constraint_kd_in_the_student_loss(teachers, tmp_path):
    """warm-up step of a range student with range teachers: the constraint term is the only difference to the same step
    with kd_constraint off (same student, same teacher scores), it reaches the student's field heads."""
    out = {}
    for on in (False, True):
        c = TL.cfg2(tmp_path / f"on{int(on)}", v4=dict(V43, kd_constraint={"enabled": on}), kd_score_keys=["nc", "dac"],
                    lat_kd={"start_mb": 0}, teacher_det_run=str(teachers["T"]), teacher_map_run=str(teachers["M"]))
        ck = O.CKE2E2(c)
        st = TLOSS._student(c)
        f, t = TL.fresh(2)
        pred = TL.fake_predictions(2, seed=0)
        loss, logs = ck.loss(st, f, t, pred)
        loss.backward()
        out[on] = (float(loss), {k: float(v) for k, v in logs.items()}, st)
    (l0, g0, _), (l1, g1, st1) = out[False], out[True]
    assert "v4/cn_occ" not in g0 and g1["v4/cn_det_rows"] > 0 and g1["v4/cn_map_rows"] > 0
    assert abs(g1["ck2/bce"] - g0["ck2/bce"]) < 1e-6 and abs(g1["ck2/kd_score"] - g0["ck2/kd_score"]) < 1e-6
    assert abs((l1 - l0) - g1["ck2/wt_kd_constraint"]) < 1e-4
    assert abs(g1["ck2/wt_kd_constraint"] - (0.5 * g1["v4/cn_occ"] + 0.5 * g1["v4/cn_margin"])) < 1e-5
    for head in (st1.v4_readout.blk_head, st1.v4_readout.marg_head):
        assert float(head[0].weight.grad.abs().sum()) > 0


# ----------------------------------------------------------------------------------------------- kd_score_scope near
def _ref_one(objs, v_e=10.0, sdf=None):
    """ref_* of one token: straight route centerline along +x (0.25 m vertices from -30 m), objects = list of (x, y,
    heading, vx, vy, L, cls) observed at every keyframe."""
    A = max(len(objs), 1)
    kf = torch.zeros(1, 800, 11, 6)
    first = torch.zeros(1, 800, 6)
    meta = torch.zeros(1, 800, 5, dtype=torch.int16)
    for a, (x, y, h, vx, vy, L, cls) in enumerate(objs):
        t = torch.arange(11) * 0.5
        kf[0, a, :, 0], kf[0, a, :, 1], kf[0, a, :, 2] = x + vx * t, y + vy * t, h
        kf[0, a, :, 3], kf[0, a, :, 4], kf[0, a, :, 5] = vx, vy, 1.0
        first[0, a] = torch.tensor([L, 2.0, h, vx, vy, 0.0])
        meta[0, a] = torch.tensor([cls, int(cls in (0, 1, 2)), 0, 10, 0], dtype=torch.int16)
    cl = torch.zeros(1, 1536, 2)
    n = 1100
    cl[0, :n, 0] = -30.0 + 0.25 * torch.arange(n)
    clv = torch.zeros(1, 1536, dtype=torch.bool)
    clv[0, :n] = True
    st = torch.zeros(1, 8)
    st[0, 4] = v_e
    ref = {"ref_obj_kf": kf, "ref_obj_first": first, "ref_obj_meta": meta, "ref_obj_n": torch.tensor([A], dtype=torch.int32),
           "ref_obj_n_kf": torch.tensor([11], dtype=torch.int16), "ref_cl_xy": cl, "ref_cl_valid": clv,
           "ref_gt_ok": torch.tensor([True]), "ref_sdf": torch.zeros(1, 320, 256) if sdf is None else sdf}
    return ref, st


def _cat(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in a}


def test_lead_ttc0_rule():
    objs = [(20.0, 0.5, 0.0, 2.0, 0.0, 4.5, 0),            # the lead: gap 20 - 2.25 - 4.049, closing 8 m/s
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1),            # a pedestrian (not a lead class)
            (-8.0, 0.0, 0.0, 9.0, 0.0, 4.5, 0),            # behind (gap < -1)
            (14.0, 3.0, 0.0, 0.0, 0.0, 4.5, 0),            # 3 m lateral
            (16.0, 0.0, 2.0, 0.0, 0.0, 4.5, 0),            # crossing (heading 115 deg)
            (35.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2)]            # a stopped bicycle farther ahead
    r1, s1 = _ref_one(objs)
    r2, s2 = _ref_one([(20.0, 0.5, 0.0, 12.0, 0.0, 4.5, 0)])          # opening gap: no closing lead
    r3, s3 = _ref_one(objs)
    r3["ref_gt_ok"] = torch.tensor([False])                            # no GT
    ref, st = _cat(_cat(r1, r2), r3), torch.cat([s1, s2, s3])
    ttc, gt = O.lead_ttc0(ref, st)
    assert gt.tolist() == [True, True, False]
    assert abs(float(ttc[0]) - (20.0 - 2.25 - 4.049) / 8.0) < 1e-6
    assert float(ttc[1]) == O.TTC0_NONE and float(ttc[2]) == O.TTC0_NONE


@have
def test_lead_ttc0_real_tokens_and_parent_sdf():
    """the fixture tokens against CK_DATA/lead/navtrain_train.parquet (the lead_split.py rule), and the parent's min
    executed corner SDF against range_targets.executed_slack's DAC slack on the official SDF."""
    from navsim.agents.para_ssr.ck.v4 import lqr as LQR
    from navsim.agents.para_ssr.ck.v4 import range_targets as RT
    from navsim.agents.para_ssr.refiner.corridor import corridor_geometry
    f, t = TL.fresh(2)
    st = f["status_feature"].float()
    ttc, gt = O.lead_ttc0(t, st)
    lp = CK / "lead" / "navtrain_train.parquet"
    if lp.is_file():
        df = pd.read_parquet(lp).set_index("token")
        for i, tok in enumerate(TL.batch()["tokens"]):
            if tok in df.index and bool(gt[i]):
                r = df.loc[tok]
                want = float(r["ttc0"]) if int(r["has_lead"]) == 1 else O.TTC0_NONE
                assert (float(ttc[i]) < O.NEAR_TTC_S) == (want < O.NEAR_TTC_S), (tok, float(ttc[i]), want)
    pred = TL.fake_predictions(2, seed=2)
    par = pred["trajectory_anchors"][None, :12].expand(2, 12, 8, 3).float().contiguous()
    m = O.parent_min_corner_sdf(t, st, par)
    assert m.shape == (2, 12)
    ex = LQR.simulate(par.reshape(-1, 8, 3), torch.stack([st[:, 4], st[:, 6]], -1).repeat_interleave(12, 0),
                      dtype=torch.float32)
    geom = corridor_geometry(par.reshape(-1, 8, 3), version=2)
    ref = RT.executed_slack(ex, geom, torch.zeros(24, 48, 17, M, dtype=torch.bool), field=t["ref_sdf"].float(),
                            sdf_index=torch.arange(2).repeat_interleave(12))
    want = ref["slack_dac"].reshape(2, 12)
    fin = torch.isfinite(m) & bool(t["ref_gt_ok"].all())
    assert bool(fin.any())
    assert torch.allclose(m[fin], want[fin].float(), atol=1e-4)


def test_score_kd_near_equals_all_and_restricts():
    g = torch.Generator().manual_seed(1)
    s = torch.randn(3, 6, 2, generator=g)
    p = torch.rand(3, 6, 2, generator=g)
    ok = torch.rand(3, 6, generator=g) > 0.2
    from navsim.agents.para_ssr.ck import losses as L
    full = O.score_kd_near(s, p, ok, ["nc", "dac"], torch.ones(3, dtype=torch.bool), torch.ones(3, 6, dtype=torch.bool))
    assert torch.allclose(full, L.score_kd_bce(s, p, ok), atol=1e-7)
    nc_tok = torch.tensor([True, False, True])
    dac = torch.rand(3, 6, generator=g) > 0.5
    got = O.score_kd_near(s, p, ok, ["nc", "dac"], nc_tok, dac)
    e = 0.5 * (L.score_kd_bce(s[..., :1], p[..., :1], ok & nc_tok[:, None])
               + L.score_kd_bce(s[..., 1:], p[..., 1:], ok & dac))
    assert torch.allclose(got, e, atol=1e-7)


@have
def test_kd_score_scope_near_in_the_loss(tmp_path, monkeypatch):
    seen = {}
    fn = O.score_kd_near

    def spy(*a, **k):
        seen["args"] = a
        return fn(*a, **k)
    monkeypatch.setattr(O, "score_kd_near", spy)
    res = {}
    for scope in ("all", "near"):
        c = TL.cfg2(tmp_path / scope, v4={"kd_score_scope": scope})
        ck = O.CKE2E2(c)
        st = TLOSS._student(c)
        f, t = TL.fresh(2)
        t["ref_gt_ok"] = torch.tensor([True, False])                 # token 1 without GT: excluded from nc / dac KD
        loss, logs = ck.loss(st, f, t, TL.fake_predictions(2, seed=0))
        res[scope] = {k: float(v) for k, v in logs.items()}
    a, n = res["all"], res["near"]
    assert "v4/kd_near_nc_frac" in n and "v4/kd_near_nc_frac" not in a and n["v4/kd_near_gt_frac"] == 0.5
    assert abs(a["ck2/bce"] - n["ck2/bce"]) < 1e-6
    nc_tok, dac = seen["args"][4], seen["args"][5]
    assert not bool(nc_tok[1]) and not bool(dac[1].any())


# ----------------------------------------------------------------------------------------------- computed edits
# the untrained range student (seed 0, score prior 'none'): its random B / U logits block about half of the cells, so the
# moving fixture token's parents need longitudinal edits and the bisection finds them (the slow token: v0 < 2 m/s, none)


@have
def test_computed_edits_recorded_labelled_and_in_the_label_set(tmp_path):
    c, ck, _, _ = TLOSS._onpolicy(tmp_path, lat_kd={"start_mb": 0}, v4=V43, rec_chunk_tokens=1)
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target, with_edits=True, with_pref=True)
    assert O.CKE2E2(c).labels().with_edits                         # the config's own store asks for the edit files
    st = TLOSS._student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 5, c.rec_chunk_tokens)
    f, t = TL.fresh(2, rows=[3, 7])
    assert float(f["status_feature"][0, 4]) < 2.0 < float(f["status_feature"][1, 4])     # one slow, one moving token
    loss, logs = ck.loss(st, f, t, TL.fake_predictions(2, seed=3))
    assert torch.isfinite(loss) and "v4/awr" not in logs                           # no corrector: no AWR term
    assert float(logs["v4/edit_rec_lon_frac"]) > 0
    ck.recorder.close_epoch()
    recs = {}
    for ch in ck.recorder.chunks:
        d = D2.read_rec2_chunk(Path(c.io_dir) / "rec" / "ep005" / "r0" / ch)
        for i, r in enumerate(d["row"]):
            recs[int(r)] = {k: d[k][i] for k in ("traj", "edit_traj", "edit_src", "edit_valid")}
    assert set(recs) == {3, 7}
    assert not recs[3]["edit_valid"].any()                                         # v0 < 2 m/s: no edit recorded
    r7 = recs[7]
    assert r7["edit_valid"].any() and bool((r7["edit_src"] % 6 == 0).all())        # sources = identity columns
    for j in np.flatnonzero(r7["edit_valid"]):                                     # an edit, not its source
        assert not np.array_equal(r7["edit_traj"][j], r7["traj"][int(r7["edit_src"][j])])
    # labeler2 scores the 96 + 8 of the chunk (fake scorer) into the generation's edit files
    tok = TLAB._tokens(tmp_path, n=16)
    assert TLAB._lab(Path(c.io_dir), tok, n_rows=16).run() == 0
    g = D2.open_generation2(c.io_dir, 5, 16, "r", optional=D2.GEN2_EDIT_FILES)
    assert np.array_equal(np.asarray(g["edit_ok"][7]), r7["edit_valid"]) and not np.asarray(g["edit_ok"][3]).any()
    # next epoch: the edits join the critic's BCE label set with their official labels
    ck.epoch, ck.epoch_frac = 6, 6.2
    ck.recorder = None
    TLOSS._spy(ck, "current_batch")
    loss6, logs6 = ck.loss(st, f, t, TL.fake_predictions(2, seed=3))
    S = ck._spy["current_batch"]
    assert torch.isfinite(loss6) and S["extra"].shape == (2, 48 + D2.EDIT_K, 8, 3)
    ok_e = S["ok_lab"][:, 48:]
    assert torch.equal(ok_e[1], torch.from_numpy(r7["edit_valid"])) and not bool(ok_e[0].any())
    j = int(np.flatnonzero(r7["edit_valid"])[0])
    from navsim.agents.para_ssr.ck.ep_target import ck_targets
    y = ck_targets(np.asarray(g["edit_labels"][7], np.float32), c.ep_target)
    assert torch.allclose(S["y"][1, 48 + j], torch.from_numpy(y[j]).float())
    assert torch.equal(S["extra"][1, 48 + j], torch.from_numpy(r7["edit_traj"][j]))


@have
def test_infer_computed_edits_and_infer_check(tmp_path):
    from tools.ck.e2e2 import eval_e2e2 as E2
    c = TL.cfg2(tmp_path, v4=V43)
    st = TLOSS._student(c).eval()
    ck = O.CKE2E2(c)
    f, _ = TL.fresh(2)
    pred = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in TL.fake_predictions(2, seed=3).items()}
    with torch.no_grad():
        res = ck.infer(st, f, pred)
    assert res["ck2_edit_traj"].shape == (2, 8, 8, 3) and res["ck2_edit_kind"].dtype == torch.int8
    assert bool((res["ck2_edit_src"] % 6 == 0).all())
    assert not bool((res["ck2_edit_kind"][0] > 0).any()) and bool((res["ck2_edit_kind"][1] > 0).any())
    # selectable = an actual edit of an eligible parent: no-edit slots (kind 0, the source bitwise) are not
    assert torch.equal(res["ck2_edit_valid"], res["ck2_edit_kind"] > 0) and not bool(res["ck2_edit_valid"][0].any())
    k0 = res["ck2_edit_kind"] == 0
    src_traj = torch.gather(res["ck2_cand96"], 1, res["ck2_edit_src"][..., None, None].expand(-1, -1, 8, 3))
    assert bool(k0.any()) and torch.equal(res["ck2_edit_traj"][k0], src_traj[k0])
    # the writer stores every edit key; infer_check replays computed_edits on the stored BEV grids
    P = E2.Paths(tmp_path / "run", "navtrain_val")
    tdf = pd.DataFrame({"token": ["a", "b"], "log": ["l", "l"], "city": ["c", "c"]})
    w = E2.EvalRootWriter2(P, tdf)
    npd = {k: v.numpy() for k, v in res.items()}
    w.write([0, 1], npd, f["status_feature"].numpy(), np.zeros((2, 8, 3), np.float32), np.zeros((2, 8, 3), np.float32))
    w.flush()
    assert np.array_equal(np.load(P.infer / "edit_kind.npy"), res["ck2_edit_kind"].numpy())
    P.ck_run.mkdir(parents=True, exist_ok=True)
    cfg = dict(arm="S", seed=0, v4=st.v4.to_dict(), corridor_version=2)
    save_ck(P.ck_run / "ckpt_last.pt", st, cfg)
    (P.ck_run / "config.json").write_text(json.dumps(cfg))
    bevg = pred["bev_embed"].transpose(1, 2).reshape(2, 256, 50, 100)
    np.savez(P.infer / "check_bev.npz", rows=np.arange(2), bev=bevg.numpy())
    chk = E2.stage_infer_check(P, SimpleNamespace(gpus=""))
    assert chk["pass"] and "edit_traj" in chk["max_abs_err"] and "edit_score_logit" in chk["max_abs_err"]


# ----------------------------------------------------------------------------------------------- eval 96 + K
def _add_edits(root: Path, run: str, split: str, n: int = 40):
    """K = 3 edits per token: slot 0 = a perfect edit (PDMS 1, scored 8) of column 0; slot 1 = scored 9 but unlabelled;
    slot 2 = a filler (edit_valid False) scored 10."""
    inf, lb = root / "infer" / run / split, root / "labels" / split / f"edit_{run}"
    lb.mkdir(parents=True, exist_ok=True)
    K = 3
    sl = np.full((n, K, 5), -2.0, np.float32)
    sl[:, 0], sl[:, 1], sl[:, 2] = 8.0, 9.0, 10.0
    np.save(inf / "edit_score_logit.npy", sl)
    np.save(inf / "edit_src.npy", np.zeros((n, K), np.int16))
    v = np.ones((n, K), bool)
    v[:, 2] = False
    np.save(inf / "edit_valid.npy", v)
    np.save(inf / "edit_kind.npy", np.tile(np.array([1, 2, 0], np.int8), (n, 1)))
    L = Cn.LBL
    lab = np.full((n, K, 9), 0.5, np.float32)
    lab[..., L["nc"]] = lab[..., L["ttc"]] = 1.0
    lab[:, 0, L["pdms"]] = 1.0
    lab[:, 2, L["pdms"]] = 1.0
    ok = np.ones((n, K), bool)
    ok[:, 1] = False
    np.save(lb / "labels.npy", lab)
    np.save(lb / "ok.npy", ok)


def test_eval_96k_val_frozen_beta_navtest(tmp_path):
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "ck2e2e_fake", tmp_path / "lead"
    TSEL._fake_root(root, run, "navtrain_val", seed=0)
    TSEL._fake_root(root, run, "navtest", seed=1)
    base = EV.run_eval(root, run, "navtrain_val", out=tmp_path / "noedits", n_boot=50, lead_root=lead)
    assert "edits" not in base and "best_edits" not in base and "best_edits" not in base["eval_spec"]
    _add_edits(root, run, "navtrain_val")
    m = EV.run_eval(root, run, "navtrain_val", n_boot=50, lead_root=lead)
    assert m["n_eval"] == base["n_eval"]                                     # the 96 token mask, unchanged
    rows = {r["key"]: r for r in m["rows"]}
    for r in base["rows"]:                                                   # every 96-only row on the same tokens
        assert rows[r["key"]]["pdms"] == r["pdms"]
    assert m["best"] == base["best"] and m["best"]["pdms"] == pytest.approx(0.9)
    be = m["best_edits"]
    assert be["pdms"] == pytest.approx(1.0) and be["key"].startswith("sel+edits") and m["representative"] == be["key"]
    hr = rows[be["key"]]
    assert hr["edit_pick_rate"] == pytest.approx(1.0) and hr["frac_lat"] == 0.0     # the edit, no lateral on top
    assert hr["edit_pick_kind"]["lon"] == pytest.approx(1.0)                         # slot 0 (kind lon) only
    e = m["edits"]
    assert e["n_edit_unlabelled"] == m["n_eval"] and e["n_edit_usable"] == m["n_eval"]   # slot 1 unlabelled, 2 filler
    assert e["same_tokens_96"] == hr["key_96_same"] and e["best_96"] == m["best"]["key"]
    assert e["d_vs_96_best"]["mean"] == pytest.approx(0.1, abs=1e-6)
    assert m["eval_spec"]["best_edits"] == be and m["eval_spec"]["edits_k"] == 3
    vm = root / "eval" / run / "navtrain_val" / "metrics.json"
    # navtest restores both choices; the headline is the 96 + K row with the val beta / variant set
    with pytest.raises(SystemExit, match="96 \\+ K"):                         # navtest root without edits
        EV.run_eval(root, run, "navtest", fix_from=str(vm), n_boot=50, lead_root=lead)
    _add_edits(root, run, "navtest")
    t = EV.run_eval(root, run, "navtest", fix_from=str(vm), n_boot=50, lead_root=lead)
    assert t["fixed_edits_from_val"] == be and t["representative"] == be["key"] and "best_edits" not in t
    keys = {r["key"] for r in t["rows"]}
    assert be["key"] in keys and m["best"]["key"] in keys and hr["key_96_same"] in keys
    assert t["edits"]["d_vs_96_same"]["n"] == t["n_eval"]
    pt = pd.read_parquet(root / "eval" / run / "navtest" / "per_token.parquet")
    assert (pt["sel_col"] == 96).all() and np.allclose(pt["pdms_sel"], 1.0)
    # a val result without edits cannot fix a navtest eval root with edits
    with pytest.raises(SystemExit, match="96 \\+ K"):
        EV.run_eval(root, run, "navtest", fix_from=str(tmp_path / "noedits" / "metrics.json"), n_boot=50,
                    lead_root=lead, out=tmp_path / "nt2")


def test_select_edits_rules():
    from tools.ck.e2e2 import eval_ck2 as EV
    n = 3
    v2f = np.tile(np.linspace(1, 0, 16), (n, 1))
    im = np.full((n, 16), 1 / 16)
    p96 = np.full((n, 96, 5), 0.6)
    valid = np.ones((n, 96), bool)
    pe = np.full((n, 2, 5), 0.6)
    pe[0, 1] = 0.99                                                           # better than every pool column
    src = np.array([[0, 6], [0, 6], [0, 6]])
    use = np.ones((n, 2), bool)
    use[2, 1] = False
    pe[2, 1] = 0.99
    idx = EV.select_edits(v2f, im, p96, valid, pe, src, use, 1.0)
    base = EV.S2.select96(v2f, im, p96, valid, 1.0)
    assert idx[0] == 97 and idx[1] == base[1] and idx[2] == base[2]         # ties -> the pool; unusable edit skipped


def _copy_edits(root: Path, run: str, split: str, cand96: np.ndarray, src: np.ndarray, eps_traj: float,
                d_logit: float, valid: bool, kind: int = 0):
    """K = len(src) edits of the same source columns src on every token: edit_traj = cand96[src] (+ eps_traj), score =
    the pool's logit of the source + d_logit, labels = the source's POOL labels (an edit is submitted without the
    lateral head), edit_valid = valid, edit_kind = kind (0 = the rule's no-edit slot)."""
    pk, inf, lb = root / "packed" / split, root / "infer" / run / split, root / "labels" / split
    n, K = cand96.shape[0], len(src)
    np.save(pk / "cand96.npy", cand96)
    np.save(inf / "edit_traj.npy", (cand96[:, src] + np.float32(eps_traj)).astype(np.float32))
    np.save(inf / "edit_src.npy", np.tile(np.asarray(src, np.int16), (n, 1)))
    np.save(inf / "edit_score_logit.npy", (np.load(inf / "score_logit.npy")[:, src] + np.float32(d_logit)))
    np.save(inf / "edit_valid.npy", np.full((n, K), valid))
    np.save(inf / "edit_kind.npy", np.full((n, K), kind, np.int8))
    d = lb / f"edit_{run}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "labels.npy", np.load(lb / f"pool96_{run}" / "labels.npy")[:, src])
    np.save(d / "ok.npy", np.ones((n, K), bool))


def test_eval_no_edit_copies_never_selected(tmp_path):
    """review (wiring, major): a no-edit slot is its source column bitwise and its re-score ties the pool's only up to
    GEMM batch-shape noise (here: +1e-6 on the logits).  Whether the dump marks it unselectable (edit_valid False, the
    fixed CKE2E2.infer) or not (an older dump, edit_valid True), the 96 + K rows equal the 96 rows; a real edit (not a
    copy) with the same scores IS picked (the test bites: the planted column 3 is then submitted without the lateral
    head)."""
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "ck2e2e_fake", tmp_path / "lead"
    TSEL._fake_root(root, run, "navtrain_val", seed=0)
    n = 40
    c96 = np.random.default_rng(3).normal(size=(n, 96, 8, 3)).astype(np.float32)
    src = np.array([0, 3, 6])

    def ev(name, **kw):
        _copy_edits(root, run, "navtrain_val", c96, src, **kw)
        m = EV.run_eval(root, run, "navtrain_val", out=tmp_path / name, n_boot=20, lead_root=lead)
        rows = {r["key"]: r for r in m["rows"]}
        pairs = [(r, rows[r["key_96_same"]]) for r in m["rows"] if r["variant"] == "sel_edits"]
        assert len(pairs) == sum(r["variant"] == "sel" for r in m["rows"])
        return m, pairs

    m, pairs = ev("real", eps_traj=1e-3, d_logit=1e-6, valid=True, kind=1)                # a real edit: picked
    assert m["edits"]["n_edit_dup"] == 0 and any(r["edit_pick_rate"] > 0 and r["pdms"] != r96["pdms"]
                                                 for r, r96 in pairs)
    for name, valid in (("old_dump", True), ("fixed_dump", False)):                        # copies: never picked
        m, pairs = ev(name, eps_traj=0.0, d_logit=1e-6, valid=valid)
        e = m["edits"]
        assert e["n_edit_usable"] == 0 and e["n_edit_dup"] == (3 * m["n_eval"] if valid else 0)
        for r, r96 in pairs:
            assert r["edit_pick_rate"] == 0.0 and r["pdms"] == r96["pdms"] and r["frac_lat"] == r96["frac_lat"]
        assert m["best_edits"]["pdms"] == m["best"]["pdms"] and e["d_vs_96_best"]["mean"] == 0.0


def test_eval_navtest_refuses_other_edit_k(tmp_path):
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "ck2e2e_fake", tmp_path / "lead"
    TSEL._fake_root(root, run, "navtrain_val", seed=0)
    TSEL._fake_root(root, run, "navtest", seed=1)
    _add_edits(root, run, "navtrain_val")
    EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead)
    _add_edits(root, run, "navtest")
    inf, lb = root / "infer" / run / "navtest", root / "labels" / "navtest" / f"edit_{run}"
    for f in [inf / f"edit_{k}.npy" for k in ("score_logit", "src", "valid", "kind")] + [lb / "labels.npy",
                                                                                          lb / "ok.npy"]:
        np.save(f, np.load(f)[:, :2])                                                      # K = 2 on navtest
    with pytest.raises(SystemExit, match="K = 3 edits"):
        EV.run_eval(root, run, "navtest", fix_from=str(root / "eval" / run / "navtrain_val" / "metrics.json"),
                    n_boot=20, lead_root=lead)


def test_near_scope_refuses_missing_ref_keys():
    S = {"cand": torch.zeros(2, 4, 8, 3)}
    t = {k: torch.zeros(2) for k in O.NEAR_REF_KEYS if k != "ref_sdf"}
    with pytest.raises(ValueError, match="ref_sdf"):
        O.CKE2E2._near_scope(t, torch.zeros(2, 8), S, {})


def test_infer_select_edits_config_rules():
    sel = {"beta": 1.0, "set": "all", "lat_mode": "on"}
    base = {"enabled": True, "io_dir": "/tmp/x", "v4": V43}
    assert O.CKE2E2Config.from_any({"infer_select": sel}).infer_select == sel                 # old dict exactly
    assert O.CKE2E2Config.from_any({"infer_select": dict(sel, edits="true")}).infer_select == dict(sel, edits=True)
    with pytest.raises(ValueError, match="needs exactly"):
        O.CKE2E2Config.from_any({"infer_select": dict(sel, edit=True)})
    with pytest.raises(ValueError, match="emits edits"):                                     # explicit choice needed
        O.CKE2E2Config.from_any(dict(base, infer_select=sel)).validate(None)
    with pytest.raises(ValueError, match="needs a student with edits"):
        O.CKE2E2Config.from_any({"enabled": True, "io_dir": "/tmp/x", "infer_select": dict(sel, edits=True)}
                                ).validate(None)


@have
def test_infer_select_edits_96k(tmp_path, monkeypatch):
    sel = {"beta": 1.0, "set": "all", "lat_mode": "on"}
    f, _ = TL.fresh(2)
    pred = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in TL.fake_predictions(2, seed=3).items()}
    c = TL.cfg2(tmp_path, v4=V43, infer_select=sel)
    st = TLOSS._student(c).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="emits edits"):                    # infer_select.edits unset
        O.CKE2E2(c).infer(st, f, pred)
    with torch.no_grad():
        r96 = O.CKE2E2(TL.cfg2(tmp_path, v4=V43, infer_select=dict(sel, edits=False))).infer(st, f, pred)
        res = O.CKE2E2(TL.cfg2(tmp_path, v4=V43, infer_select=dict(sel, edits=True))).infer(st, f, pred)
    assert int(r96["ck2_sel_idx"].max()) < 96                                                # edits false: 96 only
    npf = lambda k: res[k].float().numpy()                                                   # noqa: E731
    want = O.select96_edits(npf("ck2_v2_final"), npf("ck2_v2_im"), torch.sigmoid(res["ck2_score_logit"]).numpy(),
                            res["ck2_valid96"].numpy(), torch.sigmoid(res["ck2_edit_score_logit"]).numpy(),
                            res["ck2_edit_src"].numpy(), res["ck2_edit_valid"].numpy(), 1.0)
    assert res["ck2_sel_idx"].tolist() == want.tolist()
    # the submitted trajectory: an edit pick = the edit itself, a pool pick = its laterally corrected column
    j = int(torch.nonzero(res["ck2_edit_valid"][1])[0])
    monkeypatch.setattr(O, "select96_edits", lambda *a, **k: np.array([5, 96 + j]))
    with torch.no_grad():
        forced = O.CKE2E2(TL.cfg2(tmp_path, v4=V43, infer_select=dict(sel, edits=True))).infer(st, f, pred)
    assert forced["ck2_sel_idx"].tolist() == [5, 96 + j]
    assert torch.equal(forced["ck2_traj"][0], forced["ck2_lat_traj"][0, 5])
    assert torch.equal(forced["ck2_traj"][1], forced["ck2_edit_traj"][1, j])


# ----------------------------------------------------------------------------------------------- v4_3 arm / launch
def _set_file(path: Path, teachers, extra=()):
    lines = [f"teacher_det_run={teachers['T']}", f"teacher_map_run={teachers['M']}", "kd_calib.enabled=false",
             "teacher_ep_check=false", *extra]
    path.write_text("\n".join(lines) + "\n")
    return path


@have
def test_v4_3_arm_overrides_and_refusals(teachers, tmp_path):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from tools.ck.e2e2 import launch_util2 as L2
    sets = L2.LU.read_lines(str(L2.V4_3_SET))
    base = L2.load_ck2_yaml()
    with pytest.raises(L2.LaunchError, match="unknown"):                     # the plain parser: optional keys unknown
        L2.LU.apply_sets(base, sets)
    eff = L2.apply_sets2(base, sets)
    assert L2.arm_of(eff) == "v4_3" and eff["v4"]["edits"] == "computed" and eff["v4"]["kd_constraint"]["enabled"]
    assert base["v4"] == O._d_v4() and L2.apply_sets2(base, L2.LU.read_lines(str(L2.V4_SET)))["v4"].keys() == \
        O._d_v4().keys()                                                     # older arms: no optional key declared
    with pytest.raises(L2.LaunchError, match="PLACEHOLDER"):
        L2.build_overrides2(str(tmp_path / "ck2e2e_v43"), ["0", "1"], ck_sets=sets, wandb="disable")
    r2t = dict(teachers, T=teachers["T_r2"], M=teachers["M_r2"])            # the arm is R2: R2 teachers
    own = L2.LU.read_lines(str(_set_file(tmp_path / "t.set", r2t)))
    ov, ck, tj = L2.build_overrides2(str(tmp_path / "ck2e2e_v43"), ["0", "1"], ck_sets=sets + own, wandb="disable")
    c = O.CKE2E2Config.from_any(ck)
    assert c.v4_net() == V4Cfg.range_arm(edits="computed", rg_read="any4", rg_stopped_aware=True)
    assert c.kd_score_keys == ["nc", "dac"] and c.kd_constraint["occ_channels"] == "B"
    assert c.v4["kd_parent_only"] and not c.v4["corrector"] and c.bev_kd["enabled"] and c.kd_score_scope == "all"
    assert tj["kd_constraint"]["det"]["readout"] == "range" and tj["kd_constraint"]["student_grid"]["ev_tbins"] == 20
    parsed = OverridesParser.create().parse_overrides([o for o in ov if o.startswith("++agent.config.ck_e2e2.")])
    d: dict = {}
    for p in parsed:
        key = p.key_or_group.split("agent.config.ck_e2e2.", 1)[1].split(".")
        cur = d
        for k in key[:-1]:
            cur = cur.setdefault(k, {})
        cur[key[-1]] = p.value()
    assert O.CKE2E2Config.from_any(d).to_dict() == c.to_dict()
    # constraint KD refuses mean teachers and a teacher on another grid
    for det in (TL.SMOKE_T, teachers["T_cv1"]):
        bad = L2.LU.read_lines(str(_set_file(tmp_path / "b.set", dict(r2t, T=det))))
        with pytest.raises(L2.LaunchError, match="range teachers|grid"):
            L2.build_overrides2(str(tmp_path / "ck2e2e_v43b"), ["0", "1"], ck_sets=sets + bad, wandb="disable")


@have
def test_train_e2e2_dry_run_arm_v4_3(teachers, tmp_path):
    sf = _set_file(tmp_path / "t.set", dict(teachers, T=teachers["T_r2"], M=teachers["M_r2"]))   # R2 arm + teachers
    env = dict(os.environ, E2E_ROOT=str(tmp_path / "root"), E2E_DRY_RUN="1", CK2_E2E_APPROVED="1", ARM="v4_3",
               RUN="dry_v43", PYTHONPATH=str(REPO))
    sh = lambda e: subprocess.run(["bash", str(REPO / "tools/ck/e2e2/train_e2e2.sh"), "start", "--gpus", "0,1"],
                                  env=e, capture_output=True, text=True, timeout=600)
    r = sh(env)
    assert r.returncode == 3 and "PLACEHOLDER" in r.stderr, r.stderr[-1500:]          # teacher paths not set
    r = sh(dict(env, E2E_CK_SET=str(sf)))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "arm v4_3" in r.stdout and "++agent.config.ck_e2e2.v4.readout=range" in r.stdout
    assert "++agent.config.ck_e2e2.v4.edits=computed" in r.stdout and "kd_constraint.enabled=true" in r.stdout
    R = tmp_path / "root" / "dry_v43"
    assert not (R / "train").exists() and (R / "launch_dryrun" / "teachers.json").is_file()


def test_run_teachers_readout_option(tmp_path):
    sh = REPO / "tools/ck/v4/run_teachers.sh"
    run = lambda **e: subprocess.run(["bash", str(sh), "both"], env=dict(os.environ, **e), capture_output=True,
                                     text=True, timeout=120)
    assert run(READOUT="bogus").returncode == 2
    assert run().returncode == 2                                                   # no approval
    if not ((TIDX / "navtrain_train").is_dir() and (TIDX / "navtrain_val").is_dir()):
        pytest.skip("time-index sidecars absent")
    r = run(V4_DRY_RUN="1", READOUT="range", V4_OUT=str(tmp_path / "out"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (tmp_path / "out").exists()                                         # a dry run writes nothing
    det = next(x for x in r.stdout.splitlines() if x.startswith("DET "))[4:].split()
    mp = next(x for x in r.stdout.splitlines() if x.startswith("MAP "))[4:].split()
    from tools.ck import train_ck2 as T2
    for args, arm, keys in ((det, "T", ["nc", "ep", "ttc", "comfort"]), (mp, "M", ["dac", "ep"])):
        a = T2.resolve_args(T2.get_parser().parse_args(args))
        assert a.arm == arm and a.v4_cfg == V4Cfg.range_arm() and a.corridor_version == 2
        assert sorted(a.score_keys) == sorted(keys) and a.run.endswith("_rng")
    rh = run(V4_DRY_RUN="1")                                                        # default: the hazard teachers
    assert rh.returncode == 0 and "--v4-readout hazard" in rh.stdout and "v4T10_haz" in rh.stdout


def test_train_ck2_v4_flags():
    from tools.ck import train_ck2 as T2
    p = T2.get_parser()
    base = ["--arm", "T", "--run-name", "x", "--device", "cpu"]
    a = T2.resolve_args(p.parse_args(base + ["--v4-readout", "range", "--v4-pref-head", "1"]))
    assert a.v4_cfg == V4Cfg.range_arm()
    a = T2.resolve_args(p.parse_args(base + ["--v4-readout", "range", "--v4-exec-model", "plan", "--v4-ev-tbin-s",
                                             "0.5", "--v4-ep-evidence", "0", "--v4-vmax", "feature"]))
    assert (a.v4_cfg.exec_model, a.v4_cfg.ev_tbins, a.v4_cfg.ev_tbin_s, a.v4_cfg.ep_evidence, a.v4_cfg.vmax) == \
        ("plan", 10, 0.5, False, "feature") and not a.v4_cfg.pref_head
    with pytest.raises(SystemExit):
        T2.resolve_args(p.parse_args(base + ["--v4-readout", "hazard", "--v4-exec-model", "lqr"]))
    a = T2.resolve_args(p.parse_args(base + ["--v4-readout", "hazard", "--v4-pref-head", "1"]))
    assert a.v4_cfg == V4Cfg(readout="hazard", pref_head=True)                     # older readouts: the old call
    assert T2.resolve_args(p.parse_args(base)).v4_cfg == V4Cfg()


def _have_real_train():
    need = [CK / "packed" / "navtrain_train" / "tokens.parquet", TIDX / "navtrain_train", TL.SMOKE_T / "norm.npz",
            TL.SMOKE_M / "norm_map.npz"]
    return all(p.exists() for p in need)


@pytest.mark.skipif(not _have_real_train(), reason="real navtrain_train data / time-index sidecars absent")
@pytest.mark.parametrize("arm", ["T", "M"])
def test_train_ck2_range_three_cpu_steps_real_data(tmp_path, arm):
    """train_ck2 --v4-readout range on real navtrain_train tokens (BEV caches, ref_* GT, time-index sidecars): 3 CPU
    steps, finite losses, the GT field aux of the arm's own keys only, the network moves, config / load_ck round trip."""
    from navsim.agents.para_ssr.ck.model import load_ck
    from tools.ck import train_ck2 as T2
    keys = "nc,ttc,ep,comfort" if arm == "T" else "dac,ep"
    norm = TL.SMOKE_T if arm == "T" else TL.SMOKE_M
    args = ["--arm", arm, "--run-name", f"rng3{arm}", "--out-root", str(tmp_path), "--device", "cpu", "--max-steps",
            "3", "--limit-tokens", "8", "--tokens-per-batch", "2", "--workers", "0", "--v4-readout", "range",
            "--v4-pref-head", "1", "--corridor-version", "2", "--time-idx-root", str(TIDX), "--val-split", "none",
            "--train-eval-rows", "0", "--score-keys", keys, "--sur-groups", "near_mid", "--norm-from", str(norm),
            "--prior-rows", "8", "--log-every", "1", "--ckpt-every", "0"]
    a = T2.resolve_args(T2.get_parser().parse_args(args))
    run = T2.train(a)
    done = json.loads((run / "done.json").read_text())
    assert done["steps"] == 3 and done["n_skip"] == {"loss": 0, "grad": 0}
    steps = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines() if '"kind": "step"' in x]
    assert [r["step"] for r in steps] == [1, 2, 3]
    for r in steps:
        assert np.isfinite(r["loss"]) and r["rg_n"] > 0
        if arm == "T":
            assert np.isfinite(r["rg_occ_bce"]) and 0 < r["rg_rate_B"] <= r["rg_rate_U"] < 1
            assert "rg_margin_huber" not in r
        else:
            assert np.isfinite(r["rg_margin_huber"]) and r["rg_sdf_valid"] > 0.5 and "rg_occ_bce" not in r
        assert not any(k.startswith("st_bce") for k in r)                           # station BCE: hazard only
    cfg = json.loads((run / "config.json").read_text())
    assert V4Cfg.from_dict(cfg["v4"]) == V4Cfg.range_arm() and cfg["time_idx_root"] == str(TIDX)
    net, _ = load_ck(run, "last")
    fresh = CKNet(arm, seed=0, norm=load_arm_norm(run, arm), v4=V4Cfg.range_arm(), corridor_version=2)
    head = "blk_head" if arm == "T" else "marg_head"
    w1, w0 = getattr(net.v4_readout, head)[0].weight, getattr(fresh.v4_readout, head)[0].weight
    assert float((w1 - w0).abs().max()) > 0


@have
def test_teacher_r34_eval_range_teacher(teachers):
    """eval_teacher_navtest_r34 on a range teacher: teacher_forward (16 + 80 variants on their parents) == CKNet.forward
    (the sanity checks of --limit runs), finite outputs."""
    from tools.ck.e2e2 import eval_teacher_navtest_r34 as R
    from navsim.agents.para_ssr.ck.model import load_ck
    net, _ = load_ck(teachers["T"], "last")
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    cand = torch.stack([torch.stack([(5 + k) * t, 0.1 * k * t, 0 * t], -1) for k in range(16)])[None]
    var = cand[:, :, None].expand(1, 16, R.NVAR, 8, 3).reshape(1, 16 * R.NVAR, 8, 3).clone()
    var[..., 0] *= 0.9
    st = torch.zeros(1, 8)
    st[0, 1], st[0, 4] = 1, 6.0
    b = {"bev": torch.randn(1, 256, 50, 100), "cand": cand, "status": st}
    out = R.sanity_checks(net, b, var, torch.device("cpu"))
    assert out["fwd_score16_maxdiff"] < 1e-5 and out["fwd_score80_maxdiff"] < 1e-5 and out["z_lon_absmax"] == 0.0
    assert out["k_subset_score_maxdiff"] < 1e-5
    o = R.teacher_forward(net, b["bev"], cand, st, var)
    assert o["s16"].shape == (1, 16, 5) and o["s80"].shape == (1, 80, 5) and bool(torch.isfinite(o["s80"]).all())


def test_extract_v4_3_student(tmp_path):
    """eval_e2e2 extract of a v4_3 student (optional v4 network keys): load_ck rebuilds the range + computed-edits
    network strictly (eval_e2e.extract_ck alone reads the v4_2 keys only and could not)."""
    from tools.ck.e2e2 import eval_e2e2 as E2
    from navsim.agents.para_ssr.ck.model import load_ck
    c = O.CKE2E2Config.from_any({"v4": V43})
    net = CKNet("S", seed=0, v4=c.v4_net(), corridor_version=2)
    ck = tmp_path / "last.ckpt"
    torch.save({"state_dict": {f"agent.ck_student.{k}": v for k, v in net.state_dict().items()}}, ck)
    cfg = E2.extract_ck2(ck, tmp_path / "ck_run", "r", {"enabled": True, "seed": 0, "v4": dict(c.v4)})
    assert V4Cfg.from_dict(cfg["v4"]) == V4Cfg.range_arm(edits="computed") and cfg["corridor_version"] == 2
    net2, _ = load_ck(tmp_path / "ck_run", "last")
    assert net2.v4.computed_edits and net2.corridor_version == 2
