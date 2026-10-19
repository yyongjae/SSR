"""refiner v4_3 Phase 0b representation R2 + the (c) preparation (docs/refiner_v4/V4_3_SPEC.md 'Phase 0b'):
  (1) range_targets stopped-aware B (torch blocked_field + numpy twin) = the union of the two object-subset calls of
      tools/ck/v4/oracle/oracle_common.stopped_aware_B01 (synthetic exact cells, random scenes, bitwise on real
      navtrain_val tokens incl. the training ref_* path of the GT loader); stopped_mask_np / stopped_from_ref;
      gt_field_aux (= train_ck2.range_gt_aux, the stopped-aware target);
  (2) V4Cfg rg_read / rg_stopped_aware (defaults off, range only, older v4_3 dicts load; rg_read the same network,
      rg_stopped_aware + the Bp group) and the range readout's 'any4' read (hard slacks =
      range_targets.executed_slack(read='any4'), soft OR >= nearest, the prior); the Bp group (TTC reads the plain
      not-behind B, NC the stopped-aware B: routing, gradients, and on real F1 tokens with GT fields NC bitwise =
      executed_slack, TTC bitwise = the readout without R2, and close to the R2 GT tube);
  (3) train_ck2 --v4-rg-read / --v4-rg-stopped-aware (+ the resume check on a real 1-step CPU run);
  (4) run_teachers.sh READOUT=range R2=1 (v4T10_rng2 / v4M10_rng2) and its WAIT mode (fake nvidia-smi; dry runs only;
      a live chain / waiter of the tag refuses);
  (5) online2 v4.rg_read / rg_stopped_aware / rg_gt_aux (config rules, the student's GT field aux in the loss, the
      teachers' B semantics check).
CPU only:

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_r2.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
import test_ck2_e2e_loss as TLOSS  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.model import CKNet, load_arm_norm, save_ck  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import V4Cfg  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import config as C  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import range_readout as RR  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import range_targets as RT  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry, lateral_offsets  # noqa: E402
from navsim.agents.para_ssr.refiner.e2e import ego_inputs  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import RA2C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
TIDX = CK / "v4" / "time_idx"
F64 = torch.float64
R2 = V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True)
have = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
have_val = pytest.mark.skipif(not ((CK / "v3/gt_objects/navtrain_val").is_dir()
                                   and (CK / "packed/navtrain_val/cand.npy").is_file()
                                   and (REPO / "tools/ck/v4/oracle/oracle_common.py").is_file()),
                              reason="navtrain_val GT objects / packed candidates absent")


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


def _random_scene(seed, P=3, A=12, T=50):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(P, 2, generator=g, dtype=F64)
    par = torch.stack([_draft(2 + 9 * float(u[k, 0]), 0.08 * (float(u[k, 1]) - 0.5), F64) for k in range(P)])
    tt = torch.arange(T, dtype=F64) * 0.1
    c = torch.rand(A, 1, 2, generator=g, dtype=F64) * torch.tensor([45.0, 16.0], dtype=F64) \
        - torch.tensor([3.0, 8.0], dtype=F64)
    vel = (torch.rand(A, 1, 2, generator=g, dtype=F64) - 0.5) * 12
    xy = c + vel * tt[None, :, None]
    h = torch.rand(A, 1, generator=g, dtype=F64).expand(A, T) * 6.3
    LW = torch.rand(A, 1, 2, generator=g, dtype=F64).expand(A, T, 2) * torch.tensor([4.0, 1.5], dtype=F64) + 0.5
    boxes = torch.cat([xy, h[..., None], LW], -1)
    obs = torch.rand(A, T, generator=g) > 0.2
    stopped = torch.rand(A, generator=g) > 0.5
    return par, boxes, obs, stopped


# ----------------------------------------------------------------------------------------------- (1) stopped-aware B
def test_stopped_aware_exact_cells():
    """a static box overlapping the REAR of the ego box at a cell: not blocked by the not-behind proxy, blocked when the
    object is stopped / static; a moving object at the same place stays not blocked; one ahead is blocked either way."""
    par = _draft(8.0, 0.0, F64)[None]
    g = corridor_geometry(par)
    fr = RT.parent_frame(g)
    j, i = 10, 8                                                          # station 10, lateral offset 0
    ce = fr["pts"][0, j, i] + RA2C * torch.tensor([1.0, 0.0], dtype=F64)  # the cell's ego box centre
    T = 50
    box = lambda dx: torch.tensor([float(ce[0]) + dx, float(ce[1]), 0.0, 4.0, 1.8], dtype=F64).expand(1, T, 5)
    boxes = torch.cat([box(-3.0), box(+3.0)], 0)                          # behind (lon -3 < -1.3) / ahead (lon +3)
    obs = torch.ones(2, T, dtype=torch.bool)
    plain = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=0.25)
    for stp, want_behind in (([True, False], True), ([False, False], False), ([True, True], True)):
        sa = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=0.25, stopped_aware=True,
                              stopped=torch.tensor([stp]))
        assert bool(sa[0, j, i].all()) is True                            # the ahead box blocks the cell anyway
        only_behind = RT.blocked_field(fr, boxes[:1][None], obs[:1][None], tbin_s=0.25, stopped_aware=True,
                                       stopped=torch.tensor([stp[:1]]))
        assert bool(only_behind[0, j, i].any()) is want_behind
        assert torch.equal(sa | plain, sa) and torch.equal(plain, plain & sa)   # stopped-aware only adds cells
    assert not bool(RT.blocked_field(fr, boxes[:1][None], obs[:1][None], tbin_s=0.25)[0, j, i].any())
    # numpy twin, behind_m None (no filter: the flag changes nothing), argument checks
    frn = RT.frame_np(fr)
    sa_np = RT.blocked_field_np(frn, boxes.numpy(), obs.numpy(), tbin_s=0.25, stopped_aware=True,
                                stopped=np.array([True, False]))
    assert np.array_equal(sa_np, RT.blocked_field(fr, boxes[None], obs[None], tbin_s=0.25, stopped_aware=True,
                                                  stopped=torch.tensor([[True, False]])).numpy())
    nb = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=0.25, behind_m=None)
    assert torch.equal(nb, RT.blocked_field(fr, boxes[None], obs[None], tbin_s=0.25, behind_m=None, stopped_aware=True,
                                            stopped=torch.tensor([[False, False]])))
    with pytest.raises(ValueError, match="stopped mask"):
        RT.blocked_field(fr, boxes[None], obs[None], stopped_aware=True)
    with pytest.raises(ValueError, match="only read"):
        RT.blocked_field(fr, boxes[None], obs[None], stopped=torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="!= objects"):
        RT.blocked_field_np(frn, boxes.numpy(), obs.numpy(), stopped_aware=True, stopped=np.array([True]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stopped_aware_is_the_two_subset_union(seed):
    """B_sa == B(stopped objects, no not-behind) | B(moving objects, not-behind) -- the oracle's two calls -- for the
    torch field (one pass, obj_mask subsets as reference), the numpy twin (object subsets) and the LON field
    (unchanged), at 0.1 s and 0.25 s bins."""
    par, boxes, obs, stp = _random_scene(seed)
    fr = RT.parent_frame(corridor_geometry(par, version=2))
    frn = RT.frame_np(fr)
    for tb in (0.1, 0.25):
        sa, lon = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=tb, stopped_aware=True, stopped=stp[None],
                                   lon_max=True, index=torch.zeros(3, dtype=torch.long))
        ref = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=tb, behind_m=None, obj_mask=stp[None],
                               index=torch.zeros(3, dtype=torch.long)) \
            | RT.blocked_field(fr, boxes[None], obs[None], tbin_s=tb, obj_mask=~stp[None],
                               index=torch.zeros(3, dtype=torch.long))
        assert torch.equal(sa, ref)
        _, lon0 = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=tb, lon_max=True,
                                   index=torch.zeros(3, dtype=torch.long))
        assert torch.equal(lon, lon0)
        s = stp.numpy()
        np_ref = RT.blocked_field_np(frn, boxes.numpy()[s], obs.numpy()[s], tbin_s=tb, behind_m=None) \
            | RT.blocked_field_np(frn, boxes.numpy()[~s], obs.numpy()[~s], tbin_s=tb)
        np_sa, np_lon = RT.blocked_field_np(frn, boxes.numpy(), obs.numpy(), tbin_s=tb, stopped_aware=True, stopped=s,
                                            lon_max=True)
        assert np.array_equal(np_sa, np_ref) and np.array_equal(np_sa, sa.numpy())
        assert np.array_equal(np.isfinite(np_lon), torch.isfinite(lon).numpy())
        assert bool((sa & ~(lon > -math.inf)).sum() == 0)                 # B_sa within U (any overlap)
        # details output: the fine field obeys the same rule
        d = RT.blocked_field(fr, boxes[None], obs[None], tbin_s=tb, stopped_aware=True, stopped=stp[None],
                             details=True, index=torch.zeros(3, dtype=torch.long))
        assert torch.equal(d["blocked"], sa)


def test_stopped_mask_rules_and_from_ref():
    is_agent = np.array([False, True, True, True])
    vfirst = np.array([5.0, 0.0, 0.05, 0.0501])
    assert RT.stopped_mask_np(is_agent, vfirst).tolist() == [True, True, True, False]
    A = 4
    kf = torch.zeros(2, 6, 11, 6)
    first = torch.zeros(2, 6, 6)
    meta = torch.zeros(2, 6, 5, dtype=torch.int16)
    first[0, :A, GF.FI_VX] = torch.tensor([5.0, 0.0, 0.03, 0.04])
    first[0, :A, GF.FI_VY] = torch.tensor([0.0, 0.0, 0.04, 0.04])           # hypot 0.05 / 0.0566
    meta[0, :A, GF.MT_AGENT] = torch.tensor([0, 1, 1, 1], dtype=torch.int16)
    ref = {"ref_obj_kf": kf, "ref_obj_first": first, "ref_obj_meta": meta, "ref_obj_n": torch.tensor([A, 1]),
           "ref_obj_n_kf": torch.tensor([11, 11])}
    m = RT.stopped_from_ref(ref, torch.tensor([0, 1]))
    assert m.shape == (2, A) and m[0].tolist() == [True, True, True, False]
    assert m[1].tolist() == [True, False, False, False]                     # 1 object (non-agent), padding False
    boxes, _, _, _ = RT.gt_boxes_from_ref(ref, torch.tensor([0, 1]))
    assert boxes.shape[1] == m.shape[1]                                    # the same object axis


@have_val
def test_stopped_aware_bitwise_vs_oracle_on_real_val_tokens():
    """8 seeded navtrain_val tokens, v2 r34 top-16 parents (corridor v2): oracle_common.stopped_aware_B01 (0.1 s) ==
    blocked_field_np(stopped_aware) == torch blocked_field(stopped_aware) on the oracle's boxes == torch on the GT
    loader's ref_* (gt_boxes_from_ref + stopped_from_ref: the training path); stopped_from_ref == oracle stopped_mask;
    the 0.25 s field computed directly == the oracle's pooled one."""
    sys.path.insert(0, str(REPO / "tools/ck/v4/oracle"))
    import oracle_common as OC
    import pandas as pd

    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    tok = pd.read_parquet(OC.PACK / "tokens.parquet").token.astype(str).to_numpy()
    cand = np.load(OC.PACK / "cand.npy", mmap_mode="r")
    gl = GTLoader(str(CK.parent), split="navtrain_val")
    rows = np.random.default_rng(7).choice(len(tok), 8, replace=False)
    n_extra = n_cmp = 0
    for r in rows:
        gt = OC.load_gt(str(tok[r]))
        if gt is None:
            continue
        fr = RT.parent_frame(corridor_geometry(torch.as_tensor(np.asarray(cand[r], np.float64)), version=2))
        frn = RT.frame_np(fr)
        want = OC.stopped_aware_B01(frn, gt)
        stp = OC.stopped_mask(gt)
        got_np = RT.blocked_field_np(frn, gt["boxes"], gt["obs"], tbin_s=0.1, stopped_aware=True, stopped=stp)
        got_t = RT.blocked_field(fr, torch.as_tensor(gt["boxes"])[None], torch.as_tensor(gt["obs"])[None], tbin_s=0.1,
                                 stopped_aware=True, stopped=torch.as_tensor(stp)[None]).numpy()
        ref = {k: v[None] for k, v in gl.load(str(tok[r])).items()}
        boxes, obs, _, _ = RT.gt_boxes_from_ref(ref, torch.tensor([0]))
        sref = RT.stopped_from_ref(ref, torch.tensor([0]))
        A = gt["boxes"].shape[0]
        assert np.array_equal(sref[0, :A].numpy(), stp) and not bool(sref[0, A:].any())
        got_r = RT.blocked_field(fr, boxes, obs, tbin_s=0.1, stopped_aware=True, stopped=sref).numpy()
        assert np.array_equal(got_np, want) and np.array_equal(got_t, want) and np.array_equal(got_r, want)
        d25 = RT.blocked_field_np(frn, gt["boxes"], gt["obs"], tbin_s=0.25, stopped_aware=True, stopped=stp)
        assert np.array_equal(d25, OC.pool_time(want, 0.25, "any"))
        n_extra += int((want & ~RT.blocked_field_np(frn, gt["boxes"], gt["obs"], tbin_s=0.1)).sum())
        n_cmp += 1
    assert n_cmp >= 6 and n_extra > 0                                       # the rule matters on real scenes


# ----------------------------------------------------------------------------------------------- gt_field_aux
def _ref_scene(objs, n_kf=11):
    """ref_* of one token: objects (x, y, h, vx, vy, L, is_agent) observed at every keyframe, SDF 3 m everywhere."""
    A = 8
    kf = torch.zeros(1, A, 11, 6)
    first = torch.zeros(1, A, 6)
    meta = torch.zeros(1, A, 5, dtype=torch.int16)
    for a, (x, y, h, vx, vy, L, ag) in enumerate(objs):
        t = torch.arange(11) * 0.5
        kf[0, a, :, 0], kf[0, a, :, 1], kf[0, a, :, 2] = x + vx * t, y + vy * t, h
        kf[0, a, :, 3], kf[0, a, :, 4], kf[0, a, :, 5] = vx, vy, 1.0
        first[0, a] = torch.tensor([L, 1.8, h, vx, vy, 0.0])
        meta[0, a] = torch.tensor([0 if ag else 5, int(ag), 0, 10, 0], dtype=torch.int16)
    return {"ref_obj_kf": kf, "ref_obj_first": first, "ref_obj_meta": meta,
            "ref_obj_n": torch.tensor([len(objs)], dtype=torch.int32), "ref_obj_n_kf": torch.tensor([n_kf]),
            "ref_sdf": torch.full((1, 320, 256), 3.0), "ref_gt_ok": torch.tensor([True])}


def test_gt_field_aux_stopped_aware_and_train_ck2_wrapper():
    """the shared aux: the stopped-aware target adds the cells of a static box behind the ego (B rate up, U unchanged,
    the margin term untouched) and trains the Bp group on the plain B (Bp rate == the plain B rate); the target rule
    must be the readout's (a Bp group without stopped_aware, or none with it: ValueError); train_ck2.range_gt_aux ==
    range_targets.gt_field_aux with the run's V4Cfg; GT-less tokens give zero terms."""
    from tools.ck import train_ck2 as T2
    torch.manual_seed(0)
    net = CKNet("S", seed=0, v4=R2)
    cand = torch.stack([_draft(8.0), _draft(6.0, 0.02)])[None]
    ego = ego_inputs(_status(8.0))
    feat, mem = net.trunk.scene(torch.randn(1, 256, 50, 100), 1)
    haz = net.score_set(feat, mem, ego, cand)["haz"]
    assert haz["bp_logit_u"].shape == haz["occ_logit_u"].shape
    haz0 = {k: v for k, v in haz.items() if k != "bp_logit_u"}             # the same fields without the Bp group
    ref = _ref_scene([(10.0, 0.0, 0.0, 0.0, 0.0, 4.0, False),           # static, inside the corridor
                      (25.0, 1.5, 0.0, 3.0, 0.0, 4.5, True)])            # moving agent
    out = {}
    for sa, hz in ((False, haz0), (True, haz)):
        l_o, l_m, st = RT.gt_field_aux(hz, cand, ref, tbin_s=0.25, corridor_version=2, stopped_aware=sa)
        out[sa] = (l_o, l_m, st)
        a = SimpleNamespace(v4_cfg=V4Cfg.range_arm(rg_stopped_aware=sa), rg_max_parents=8, corridor_version=2)
        w = T2.range_gt_aux(hz, cand, ref, a)
        assert torch.equal(w[0], l_o) and torch.equal(w[1], l_m) and w[2] == st
    (o0, m0, s0), (o1, m1, s1) = out[False], out[True]
    assert s0["rg_n"] == s1["rg_n"] == 2 and s1["rg_rate_B"] > s0["rg_rate_B"] and s1["rg_rate_U"] == s0["rg_rate_U"]
    assert s1["rg_rate_Bp"] == s0["rg_rate_B"] and "rg_rate_Bp" not in s0          # Bp target = the plain B
    assert torch.equal(m0, m1) and not torch.equal(o0, o1)
    g = torch.autograd.grad(o1 + m1, [haz["occ_logit_u"], haz["margin_u"], haz["bp_logit_u"]], allow_unused=True)
    assert all(x is not None and float(x.abs().sum()) > 0 for x in g)
    for hz, sa in ((haz0, True), (haz, False)):
        with pytest.raises(ValueError, match="target rule must be the readout"):
            RT.gt_field_aux(hz, cand, ref, tbin_s=0.25, stopped_aware=sa)
    ref_bad = dict(ref, ref_gt_ok=torch.tensor([False]))
    lz, mz, sz = RT.gt_field_aux(haz, cand, ref_bad, tbin_s=0.25, stopped_aware=True)
    assert sz == {"rg_n": 0.0} and float(lz) == 0.0 and float(mz) == 0.0


# ----------------------------------------------------------------------------------------------- (2) config + readout
def test_config_r2_keys():
    d = V4Cfg()
    assert d.rg_read == "nearest" and d.rg_stopped_aware is False
    assert {"rg_read", "rg_stopped_aware"} <= set(C.V43_KEYS)
    assert V4Cfg.from_dict(R2.to_dict()) == R2 and R2.rg_read == "any4" and R2.rg_stopped_aware
    old = {k: v for k, v in V4Cfg.range_arm().to_dict().items() if k not in ("rg_read", "rg_stopped_aware")}
    assert V4Cfg.from_dict(old) == V4Cfg.range_arm()                      # an earlier v4_3 dict: unchanged
    assert V4Cfg.from_dict(dict(old, rg_stopped_aware="1")).rg_stopped_aware is True
    for kw in ({"rg_read": "any4"}, {"rg_stopped_aware": True}):
        for ro in ("mean", "hazard", "evidence"):
            with pytest.raises(ValueError, match="only apply to readout 'range'"):
                V4Cfg(readout=ro, **kw).validate()
    with pytest.raises(ValueError, match="rg_read"):
        V4Cfg.range_arm(rg_read="bilinear")
    assert V4Cfg.range_arm(rg_read="any4", edits="computed").computed_edits        # edits.py reads the R2 fields
    assert V4Cfg.range_arm(rg_stopped_aware=True, edits="computed").computed_edits  # (tests/test_v4_student_r2.py)
    v42 = {k: v for k, v in V4Cfg(readout="hazard").to_dict().items() if k not in C.V43_KEYS}
    with pytest.raises(ValueError, match="v4_2"):
        V4Cfg.from_dict(dict(v42, version="v4_2", rg_read="nearest"))
    # rg_read: the same network (state dict keys, shapes, seeded values)
    a = CKNet("T", seed=0, v4=V4Cfg.range_arm()).state_dict()
    b = CKNet("T", seed=0, v4=V4Cfg.range_arm(rg_read="any4")).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # rg_stopped_aware: + the Bp group (the last B / U layer 2M -> 3M outputs), nothing else changes shape
    c = CKNet("T", seed=0, v4=R2).state_dict()
    assert c.keys() == a.keys()
    diff = [k for k in a if a[k].shape != c[k].shape]
    assert sorted(diff) == ["v4_readout.blk_head.2.bias", "v4_readout.blk_head.2.weight"], diff
    assert c["v4_readout.blk_head.2.weight"].shape[0] == 60 == 3 * a["v4_readout.blk_head.2.weight"].shape[0] // 2


def _patched(net, blk_fn, D_fn):
    net.v4_readout.scene = lambda xs, Hc: {"blk": blk_fn(xs.shape[0]), "margin": D_fn(xs.shape[0])}
    return net


def _scene(net, bev, cand, par, status):
    ego = ego_inputs(status)
    feat, mem = net.trunk.scene(bev, cand.shape[0])
    return net.score_set(feat, mem, ego, cand, par, pref=net.scene_pref(mem, ego))


def _known_fields():
    par = _draft(8.0, 0.02)
    cand = torch.stack([par, _draft(3.0, 0.02), _draft(8.0, 0.02), _draft(11.0, 0.02), _draft(7.0, 0.02)])
    cand[2, :, 1] += torch.linspace(0.3, 2.6, 8)
    cand[4, :, 1] += torch.linspace(0.1, 1.5, 8)
    geom = corridor_geometry(par[None], version=1)
    s, d = geom.s[0], lateral_offsets()
    Bb = torch.zeros(1, 48, 17, 20, dtype=torch.bool)
    tb = torch.arange(20) * 0.25
    Bb[0] = (((s >= 14) & (s < 24))[:, None, None] & (d.abs() <= 1.3)[None, :, None]
             & ((tb >= 1.5) & (tb < 3.0))[None, None])
    Dg = (2.6 - d.abs())[None, None].expand(1, 48, 17).clone()
    return par, cand[None], geom, Bb, Dg


@pytest.mark.parametrize("read", ["nearest", "any4"])
def test_readout_read_rule_equals_executed_slack(read):
    """hard slack_nc / slack_ttc / crit points of the readout == range_targets.executed_slack(read=...) on the same B
    (U = B: every TTC read is B), DAC on the same D."""
    net = CKNet("S", seed=0, v4=V4Cfg.range_arm(rg_read=read)).eval()
    par, cand, geom, Bb, Dg = _known_fields()
    blk = torch.cat([torch.where(Bb, 6.0, -6.0)] * 2, -1)
    _patched(net, lambda R: blk.expand(R, *blk.shape[1:]).contiguous(), lambda R: Dg.expand(R, 48, 17).contiguous())
    with torch.no_grad():
        o = _scene(net, torch.randn(1, 256, 50, 100), cand, par[None, None].expand(1, 5, 8, 3), _status(8.0))
    h = o["haz"]
    fr = {k: v.float() for k, v in RT.parent_frame(geom).items()}
    ref = RT.executed_slack(h["exec_states"][0], fr, Bb, field=Dg, index=torch.zeros(5, dtype=torch.long), read=read)
    for k in ("slack_nc", "slack_ttc"):
        assert torch.equal(h[k][0], ref[k]), k
    assert torch.equal(h["crit_station"][0, :, :2], ref["crit_station"][:, :2])
    assert torch.equal(h["crit_time"][0, :, :2], ref["crit_time"][:, :2])
    assert torch.allclose(h["slack_dac"][0], ref["slack_dac"], atol=1e-4)
    if read == "any4":                                                     # conservative: the slow one is now caught
        near = RT.executed_slack(h["exec_states"][0], fr, Bb, field=Dg, index=torch.zeros(5, dtype=torch.long))
        assert bool((ref["slack_nc"] <= near["slack_nc"]).all()) and bool(ref["hit_nc"][1]) and not bool(near["hit_nc"][1])


def test_any4_soft_or_cells_prior_and_gradient():
    """any4_cells = range_targets._cells('any4') (+ a duplicate mask at the grid edge); with the same weights the
    any4 soft evidence is a soft OR over a superset of the nearest reads (z_any4 >= z_nearest -> lp lower); set_prior
    keeps the uniform-field NC prior; the any4 NC gradient reaches more cells than the nearest one."""
    par = _draft(8.0, 0.02)
    geom = corridor_geometry(par[None], version=2)
    fr = {k: v.float() for k, v in RT.parent_frame(geom).items()}
    xy = torch.tensor([[[0.1, 0.0], [10.3, 0.4], [20.0, -4.9], [5.0, 5.05]]])
    gc = RR.grid_coords(fr, xy)
    jn, in_, dist = RR.any4_cells(gc)
    _, _, ok, jr, ir = RT._cells(fr, xy, "any4")
    assert torch.equal(jn[ok], jr[ok]) and torch.equal(in_[ok], ir[ok])
    j_near, i_near = RR.nearest_cells(gc)
    for q in range(4):
        if bool(gc["ok"][0, q]):
            cells = {(int(a), int(b)) for a, b in zip(jn[0, q], in_[0, q])}
            assert (int(j_near[0, q]), int(i_near[0, q])) in cells        # the nearest cell is one of the 4
            assert int(dist[0, q].sum()) == len(cells)                     # each distinct cell counted once
    assert int(dist[0, 0].sum()) == 2                                      # s < ds / 2: j0 = -1 clamps onto row 0
    torch.manual_seed(0)
    st = _status([6.0, 6.0], vy=0.1)
    cand = torch.stack([torch.stack([_draft(6.0 + 0.4 * k, 0.01 * k) for k in range(4)])] * 2)
    bev = torch.randn(2, 256, 50, 100)
    lp = {}
    for read in ("nearest", "any4"):
        net = CKNet("S", seed=3, v4=V4Cfg.range_arm(rg_read=read)).eval()
        with torch.no_grad():
            lp[read] = _scene(net, bev, cand, cand, st)["haz"]["lp_nc"]
    assert bool((lp["any4"] <= lp["nearest"] + 1e-6).all()) and bool((lp["any4"] < lp["nearest"] - 1e-4).any())
    # prior: a uniform field at the prior bias gives the NC pass prior (interior reads: 41 states x 4 cells)
    for read in ("nearest", "any4"):
        net = CKNet("S", seed=0, v4=V4Cfg.range_arm(rg_read=read)).eval()
        net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])
        b = float(net.v4_readout.blk_head[-1].bias[0])
        _patched(net, lambda R: torch.full((R, 48, 17, 40), b), lambda R: torch.full((R, 48, 17), 3.0))
        c = _draft(6.0)[None, None]
        with torch.no_grad():
            o = _scene(net, torch.randn(1, 256, 50, 100), c, c, _status(6.0))
        p = float(torch.sigmoid(o["score_logit"][0, 0, 0]))
        assert abs(p - 0.97) < 0.01, (read, p)
    # gradient: the soft OR spreads over the 2 x 2 cells
    ncell = {}
    for read in ("nearest", "any4"):
        net = CKNet("S", seed=0, v4=V4Cfg.range_arm(rg_read=read))
        blk = torch.zeros(1, 48, 17, 40, requires_grad=True)
        _patched(net, lambda R: blk.expand(R, 48, 17, 40), lambda R: torch.full((R, 48, 17), 3.0))
        c = _draft(6.0, 0.02)[None, None]
        c = c + torch.tensor([0.0, 0.25, 0.0])
        o = _scene(net, torch.randn(1, 256, 50, 100), c, _draft(6.0, 0.02)[None, None], _status(6.0))
        g = torch.autograd.grad(o["score_logit"][..., 0].sum(), blk)[0]
        ncell[read] = int((g[0, ..., :20].abs().sum(-1) > 0).sum())
    assert ncell["any4"] > 2 * ncell["nearest"] > 0


# ----------------------------------------------------------------------------------------------- Bp group (TTC)
def test_bp_group_routing_and_gradients():
    """rg_stopped_aware: NC reads group 0 (the stopped-aware B), TTC reads group 2 (Bp) where it would read B (delta 0
    and |v| delta < 2 m) and U above the switch.  Synthetic fields on a straight parent: (a) B_sa blocked, Bp / U free
    -> NC fails, TTC passes; (b) Bp blocked, B_sa / U free -> NC passes, TTC fails on Bp (crit_chan 2); (c) the TTC
    slacks / critical points of [B_sa, U, Bp] == those of the readout without R2 on [Bp, U], whatever B_sa holds;
    gradients: NC only into group 0, TTC only into groups 1 / 2."""
    M = 20
    par, cand, geom, Bb, Dg = _known_fields()
    free = torch.zeros_like(Bb)
    lg = lambda b: torch.where(b, 6.0, -6.0)
    st = _status(8.0)
    pars = par[None, None].expand(1, 5, 8, 3)
    bev = torch.randn(1, 256, 50, 100)
    h = {}
    for name, cfg, groups in (("a", R2, (Bb, free, free)), ("b", R2, (free, free, Bb)),
                              ("c3", R2, (Bb | torch.roll(Bb, 3, 1), free, Bb)), ("c2", V4Cfg.range_arm(rg_read="any4"),
                                                                               (Bb, free))):
        net = CKNet("S", seed=0, v4=cfg).eval()
        blk = torch.cat([lg(g) for g in groups], -1)
        _patched(net, lambda R, blk=blk: blk.expand(R, *blk.shape[1:]).contiguous(),
                 lambda R: Dg.expand(R, 48, 17).contiguous())
        with torch.no_grad():
            h[name] = _scene(net, bev, cand, pars, st)["haz"]
    a, b = h["a"], h["b"]
    hit = lambda x, k: x[k][0] < 0
    assert bool(hit(a, "slack_nc").any()) and not bool(hit(a, "slack_ttc").any())
    assert not bool(hit(b, "slack_nc").any()) and bool(hit(b, "slack_ttc").any())
    assert bool((b["crit_chan"][0][hit(b, "slack_ttc")] == RR.CH_BP).all())
    for k in ("slack_ttc", "crit_time", "crit_station", "crit_lat"):
        x, y = h["c3"][k][0], h["c2"][k][0]
        if k.startswith("crit_"):
            x, y = x[:, 1], y[:, 1]
        assert torch.equal(x, y), k
    assert torch.equal(h["c3"]["crit_chan"][0] == RR.CH_BP, h["c2"]["crit_chan"][0] == RR.CH_B)
    # gradients of the soft evidence per channel group
    net = CKNet("S", seed=0, v4=R2)
    blk = torch.zeros(1, 48, 17, 3 * M, requires_grad=True)
    _patched(net, lambda R: blk.expand(R, 48, 17, 3 * M), lambda R: torch.full((R, 48, 17), 3.0))
    o = _scene(net, bev, cand, pars, st)
    from navsim.agents.para_ssr.ck.v4.readout import I_NC, I_TTC
    gn = torch.autograd.grad(o["score_logit"][..., I_NC].sum(), blk, retain_graph=True)[0][0]
    gt_ = torch.autograd.grad(o["score_logit"][..., I_TTC].sum(), blk)[0][0]
    grp = lambda g, q: float(g[..., q * M:(q + 1) * M].abs().sum())
    assert grp(gn, 0) > 0 and grp(gn, 1) == 0 and grp(gn, 2) == 0
    assert grp(gt_, 0) == 0 and grp(gt_, 2) > 0                          # the 8 m/s ego also reads U beyond 2 m
    assert grp(gt_, 1) > 0
    # the head and the readout refuse a field with the wrong number of groups
    with pytest.raises(ValueError, match="groups|channels"):
        _patched(net, lambda R: torch.zeros(R, 48, 17, 2 * M), lambda R: torch.full((R, 48, 17), 3.0))
        _scene(net, bev, cand, pars, st)


@pytest.mark.parametrize("arm,ck2", [("S", 3_391_755), ("T", 3_391_243), ("M", 3_391_243)])
def test_r2_param_budget(arm, ck2):
    """the Bp group (+ rg_hidden x M + M = 1,100 parameters) keeps the range critic within 0.2 % of CK2."""
    frozen = ("trunk.lon_head", "trunk.gate_head")
    n = lambda net: sum(p.numel() for k, p in net.named_parameters() if p.requires_grad and not k.startswith(frozen))
    base, r2 = n(CKNet(arm, seed=0, v4=V4Cfg.range_arm())), n(CKNet(arm, seed=0, v4=R2))
    assert r2 - base == 54 * 20 + 20 and abs(r2 - ck2) / ck2 < 0.002, (arm, base, r2, ck2)


@pytest.mark.skipif(not ((CK / "v4/analysis/range_oracle/prep/f1_sim.npy").is_file()
                         and (CK / "v3/gt_objects/navtrain_val").is_dir()), reason="Phase 0 F1 prep / GT absent")
def test_bp_group_on_real_f1_tokens():
    """10 F1 navtrain_val tokens, the R2 GT fields of their 16 parents (eval_range_teacher.r2_fields: stopped-aware B,
    LON, D) as an R2 teacher's three groups [B_sa, U = LON > -inf, Bp = LON >= -1.3] at the official F1 executed states:
    the readout's hard NC == range_targets.executed_slack(read 'any4') on B_sa bitwise (slack and hit); its TTC ==
    the readout without R2 on [Bp, U] bitwise; vs the R2 GT tube (TTC on LON from the current ego) the TTC flags agree
    on >= 98 % of the rows (the documented 2 m switch), NC everywhere; the pre-fix routing (TTC on B_sa) disagrees more."""
    sys.path.insert(0, str(REPO / "tools/ck/v4/oracle"))
    import oracle_common as OC

    from tools.ck.v4 import eval_range_teacher as E
    D = OC.Data()
    tq = torch.arange(41, dtype=F64) * 0.1
    dis = {"fix": 0, "bug": 0}
    n_rows = n_tok = n_ttc = 0
    for i in range(10):
        T = OC.token_inputs(D, i)
        if T["gt"] is None or T["sdf"] is None:
            continue
        F = E.r2_fields(T["par"], E.load_gt_store(OC.GT_OBJ, T["token"]), T["sdf"])
        X = T["X"][:96]
        idx = OC.parent_of(np.arange(96))
        tube = E.gt_tube_flags(X, F, idx)
        Bsa = torch.as_tensor(F["B"])
        LON = torch.as_tensor(F["LON"])
        U_, Bp = LON > -math.inf, LON >= RT.BEHIND_M
        Dg = torch.as_tensor(F["D"]).float()
        fr = RT.parent_frame(corridor_geometry(torch.as_tensor(T["par32"].copy()), version=2))
        X32 = torch.as_tensor(X).float()
        rows = torch.as_tensor(idx).long()
        fix = RR.field_hits(torch.cat([Bsa, U_, Bp], -1), Dg, fr, rows, X32, tq, 20, 0.25, read="any4")
        plain = RR.field_hits(torch.cat([Bp, U_], -1), Dg, fr, rows, X32, tq, 20, 0.25, read="any4")
        bug = RR.field_hits(torch.cat([Bsa, U_], -1), Dg, fr, rows, X32, tq, 20, 0.25, read="any4")
        ref = RT.executed_slack(X32, {k: v.float() for k, v in fr.items()}, Bsa, field=Dg, index=rows, read="any4")
        assert torch.equal(fix["slack_nc"], ref["slack_nc"]) and torch.equal(fix["hit_nc"], ref["hit_nc"]), i
        assert torch.equal(fix["slack_ttc"], plain["slack_ttc"]) and torch.equal(fix["hit_ttc"], plain["hit_ttc"]), i
        assert torch.equal(fix["hit_nc"], bug["hit_nc"])
        ok = T["valid96"]
        assert np.array_equal(fix["hit_nc"].numpy()[ok], tube[ok, 0])                 # NC = the R2 GT reduction
        dis["fix"] += int((fix["hit_ttc"].numpy() != tube[:, 1])[ok].sum())
        dis["bug"] += int((bug["hit_ttc"].numpy() != tube[:, 1])[ok].sum())
        n_ttc += int(tube[ok, 1].sum())
        n_rows += int(ok.sum())
        n_tok += 1
    assert n_tok >= 8 and n_ttc > 0
    assert dis["fix"] <= 0.02 * n_rows and dis["fix"] <= dis["bug"], (dis, n_rows)


# ----------------------------------------------------------------------------------------------- (3) train_ck2
def test_train_ck2_r2_flags_and_resume_rule():
    from tools.ck import train_ck2 as T2
    p = T2.get_parser()
    base = ["--arm", "T", "--run-name", "x", "--device", "cpu", "--v4-readout", "range", "--v4-pref-head", "1"]
    a = T2.resolve_args(p.parse_args(base + ["--v4-rg-read", "any4", "--v4-rg-stopped-aware", "1"]))
    assert a.v4_cfg == R2
    a0 = T2.resolve_args(p.parse_args(base))
    assert a0.v4_cfg == V4Cfg.range_arm() and a0.v4_cfg.rg_read == "nearest" and not a0.v4_cfg.rg_stopped_aware
    assert T2.resolve_args(p.parse_args(base + ["--v4-rg-stopped-aware", "1"])).v4_cfg == \
        V4Cfg.range_arm(rg_stopped_aware=True)
    for extra in (["--v4-rg-read", "any4"], ["--v4-rg-stopped-aware", "1"]):
        with pytest.raises(SystemExit):
            T2.resolve_args(p.parse_args(["--arm", "T", "--run-name", "x", "--v4-readout", "hazard"] + extra))
    # the resume rule (V4Cfg.from_dict(config.json v4) == the run's V4Cfg): an older range run resumes unchanged,
    # never with R2
    old = {k: v for k, v in V4Cfg.range_arm().to_dict().items() if k not in ("rg_read", "rg_stopped_aware")}
    assert V4Cfg.from_dict(old) == a0.v4_cfg and V4Cfg.from_dict(old) != a.v4_cfg
    assert {"v4_rg_read", "v4_rg_stopped_aware"} <= set(T2.V4_ARG_KEYS)


def _have_real_train():
    need = [CK / "packed" / "navtrain_train" / "tokens.parquet", TIDX / "navtrain_train", TL.SMOKE_T / "norm.npz"]
    return all(p.exists() for p in need)


@pytest.mark.skipif(not _have_real_train(), reason="real navtrain_train data / time-index sidecars absent")
def test_train_ck2_r2_one_cpu_step_and_resume_check(tmp_path):
    """train_ck2 --v4-readout range with R2 on real tokens: 1 CPU step (finite, the stopped-aware B target), the R2 keys
    in config.json, a --resume with another read refused, a --resume with the same flags continues."""
    from navsim.agents.para_ssr.ck.model import load_ck
    from tools.ck import train_ck2 as T2
    args = ["--arm", "T", "--run-name", "r2T", "--out-root", str(tmp_path), "--device", "cpu", "--max-steps", "1",
            "--limit-tokens", "4", "--tokens-per-batch", "2", "--workers", "0", "--v4-readout", "range",
            "--v4-pref-head", "1", "--corridor-version", "2", "--time-idx-root", str(TIDX), "--val-split", "none",
            "--train-eval-rows", "0", "--score-keys", "nc,ttc,ep,comfort", "--sur-groups", "near_mid",
            "--norm-from", str(TL.SMOKE_T), "--prior-rows", "8", "--log-every", "1", "--ckpt-every", "0"]
    r2 = ["--v4-rg-read", "any4", "--v4-rg-stopped-aware", "1"]
    run = T2.train(T2.resolve_args(T2.get_parser().parse_args(args + r2)))
    cfg = json.loads((run / "config.json").read_text())
    assert V4Cfg.from_dict(cfg["v4"]) == R2 and cfg["v4"]["rg_read"] == "any4" and cfg["v4"]["rg_stopped_aware"]
    steps = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines() if '"kind": "step"' in x]
    assert len(steps) == 1 and np.isfinite(steps[0]["loss"]) and 0 < steps[0]["rg_rate_B"] <= steps[0]["rg_rate_U"]
    assert 0 < steps[0]["rg_rate_Bp"] <= steps[0]["rg_rate_U"]                      # the Bp group is trained
    net, _ = load_ck(run, "last")
    assert net.v4 == R2
    (run / "done.json").unlink()                                            # as an interrupted run (ckpt_last kept)
    a2 = list(args)
    a2[a2.index("--max-steps") + 1] = "2"
    with pytest.raises(SystemExit, match="resume with v4"):                 # nearest read: another network rule
        T2.train(T2.resolve_args(T2.get_parser().parse_args(a2 + ["--resume", "--v4-rg-stopped-aware", "1"])))
    run2 = T2.train(T2.resolve_args(T2.get_parser().parse_args(a2 + r2 + ["--resume"])))
    assert json.loads((run2 / "done.json").read_text())["steps"] == 2


# ----------------------------------------------------------------------------------------------- (4) run_teachers.sh
SH = REPO / "tools/ck/v4/run_teachers.sh"
FAKE_SMI = """#!/bin/bash
# fake nvidia-smi: $FAKE_SMI_STATE holds 'mem, util' lines, one consumed per call (the last repeats); FAIL = exit 9
f=${FAKE_SMI_STATE:?}
n=$(cat "$f.n" 2>/dev/null || echo 0); echo $((n+1)) > "$f.n"
total=$(wc -l < "$f"); i=$(( n < total ? n + 1 : total ))
line=$(sed -n "${i}p" "$f")
[ "$line" = FAIL ] && exit 9
case "$*" in *memory.used,utilization.gpu*) echo "$line" ;; *) echo "${line%%,*}" ;; esac
"""


def _sh(env=None, which="both", timeout=120):
    return subprocess.run(["bash", str(SH), which], env=dict(os.environ, **(env or {})), capture_output=True, text=True,
                          timeout=timeout)


def _need_tidx():
    if not ((TIDX / "navtrain_train").is_dir() and (TIDX / "navtrain_val").is_dir()):
        pytest.skip("time-index sidecars absent")


def test_run_teachers_r2_args(tmp_path):
    _need_tidx()
    from tools.ck import train_ck2 as T2
    r = _sh({"V4_DRY_RUN": "1", "READOUT": "range", "R2": "1", "V4_OUT": str(tmp_path / "out")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (tmp_path / "out").exists()                                 # a dry run without WAIT writes nothing
    det = next(x for x in r.stdout.splitlines() if x.startswith("DET "))[4:].split()
    mp = next(x for x in r.stdout.splitlines() if x.startswith("MAP "))[4:].split()
    for args, arm, name in ((det, "T", "v4T10_rng2"), (mp, "M", "v4M10_rng2")):
        a = T2.resolve_args(T2.get_parser().parse_args(args))
        assert a.arm == arm and a.run == name and a.v4_cfg == R2 and a.corridor_version == 2
    r1 = _sh({"V4_DRY_RUN": "1", "READOUT": "range"})                      # R2 unset: the v4_3 range teachers
    assert r1.returncode == 0 and "v4T10_rng " in r1.stdout and "--v4-rg-read" not in r1.stdout
    assert _sh({"V4_DRY_RUN": "1", "R2": "1"}).returncode == 2              # R2 needs READOUT=range
    assert _sh({"V4_DRY_RUN": "1", "READOUT": "range", "R2": "2"}).returncode == 2


def _wait_env(tmp_path, states, **kw):
    smi = tmp_path / "fake_smi.sh"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    st = tmp_path / "smi_state"
    st.write_text("\n".join(states) + "\n")
    Path(str(st) + ".n").unlink(missing_ok=True)
    env = {"READOUT": "range", "R2": "1", "V4_OUT": str(tmp_path / "out"), "V4_NVIDIA_SMI": str(smi),
           "FAKE_SMI_STATE": str(st), "V4_WAIT_POLL_S": "0.05"}
    env.update(kw)
    return env


def test_run_teachers_wait_mode_refusals(tmp_path):
    _need_tidx()
    env = _wait_env(tmp_path, ["0, 0"], V4_WAIT_IDLE_MIN="0.001")
    assert _sh(env).returncode == 2                                          # no approval (not a dry run)
    assert "approval" in _sh(env).stdout
    assert _sh(dict(env, V4_USER_APPROVED="1", V4_GPUS="4,5")).returncode == 2
    for bad in ("0", "-1", "x"):
        r = _sh(dict(env, V4_DRY_RUN="1", V4_WAIT_IDLE_MIN=bad))
        assert r.returncode == 2 and "V4_WAIT_IDLE_MIN" in r.stdout
    assert not (tmp_path / "out" / "teachers_rng2.wait.log").exists()


def test_run_teachers_wait_mode_idle_rule(tmp_path):
    """dry-run waiter in the foreground: busy (memory, then utilisation, then a failed query) -> waits; launches (the
    dry run) only after both GPUs were idle (< 1 GB, 0 %) for the whole window; every poll logged."""
    _need_tidx()
    states = ["5000, 0", "10, 0", "10, 0", "10, 35", "10, 0", "FAIL", "10, 0", "10, 0"]   # 2 GPUs per poll
    env = _wait_env(tmp_path, states, V4_DRY_RUN="1", V4_WAIT_IDLE_MIN="0.004")     # 0.24 s of idle
    r = _sh(env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    log = (tmp_path / "out" / "teachers_rng2.wait.log").read_text()
    lines = [x for x in log.splitlines() if " -> " in x and ("idle" in x or "busy" in x)]
    assert "5000 MiB" in lines[0] and "busy" in lines[0]
    assert "35 %" in lines[1] and "busy" in lines[1]
    assert "query failed" in lines[2] and "busy" in lines[2]
    assert all("idle" in x for x in lines[3:]) and len(lines) >= 5               # the window spans several polls
    assert "launching both" in log and "dry run (range, both): nothing launched" in log
    assert "--run-name v4T10_rng2" in log and "--v4-rg-read any4" in log and "launch exit 0" in log
    assert "git at the wait start: dry run" in log


def test_run_teachers_refuses_while_a_chain_or_waiter_of_the_tag_runs(tmp_path):
    """a live $OUT/<tag>.pid (a launched chain) refuses WAIT (also as a dry run) and a normal approved launch (exit 6,
    before the GPU check: the fake nvidia-smi reports busy, so a broken check could only exit 3, never launch); a live
    <tag>.wait.pid refuses a second waiter; a dead pid file is ignored; the foreground waiter writes its pid."""
    _need_tidx()
    out = tmp_path / "out"
    out.mkdir()
    live = subprocess.Popen(["sleep", "60"])
    try:
        (out / "teachers_rng2.pid").write_text(f"{live.pid}\n")
        env = _wait_env(tmp_path, ["5000, 0"], V4_DRY_RUN="1", V4_WAIT_IDLE_MIN="0.001")
        r = _sh(env)
        assert r.returncode == 6 and "chain is running" in r.stdout, r.stdout + r.stderr
        env_n = _wait_env(tmp_path, ["5000, 0"], V4_USER_APPROVED="1")
        r = _sh(env_n)
        assert r.returncode == 6 and "chain is running" in r.stdout, r.stdout + r.stderr
        assert not (out / "teachers_rng2.launch.json").exists() and not (out / "teachers_rng2.code.tgz").exists()
        assert _sh({"V4_DRY_RUN": "1", "READOUT": "range", "R2": "1", "V4_OUT": str(out)}).returncode == 0   # plain dry
        (out / "teachers_rng2.pid").unlink()
        (out / "teachers_rng2.wait.pid").write_text(f"{live.pid}\n")
        r = _sh(env)
        assert r.returncode == 2 and "a waiter for teachers_rng2 is running" in r.stdout
    finally:
        live.kill()
        live.wait()
    # dead pid files are ignored; the foreground (dry) waiter records its own pid
    env = _wait_env(tmp_path, ["10, 0"], V4_DRY_RUN="1", V4_WAIT_IDLE_MIN="0.001")
    (out / "teachers_rng2.pid").write_text(f"{live.pid}\n")
    r = _sh(env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "teachers_rng2.wait.pid").read_text().strip().isdigit()
    assert int((out / "teachers_rng2.wait.pid").read_text()) != live.pid


def test_run_teachers_wait_mode_gives_up_when_busy(tmp_path):
    _need_tidx()
    env = _wait_env(tmp_path, ["10, 3"], V4_DRY_RUN="1", V4_WAIT_IDLE_MIN="0.001", V4_WAIT_MAX_H="0.0002")
    r = _sh(env, timeout=60)
    assert r.returncode == 5
    log = (tmp_path / "out" / "teachers_rng2.wait.log").read_text()
    assert "gave up" in log and "launching" not in log and "nothing launched" not in log


# ----------------------------------------------------------------------------------------------- (5) online2
def test_online2_r2_and_rg_gt_aux_config():
    c = O.CKE2E2Config()
    assert c.rg_gt_aux == O._d_rg_gt_aux() and c.rg_gt_aux["enabled"] is False
    assert "rg_gt_aux" not in c.v4 and "rg_read" not in c.v4                 # optional keys absent by default
    v = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
         "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": "true",
         "rg_gt_aux": {"enabled": True, "lambda": 0.25}}
    c2 = O.CKE2E2Config.from_any({"enabled": False, "v4": v})
    c2.validate(None)
    assert c2.v4_net() == R2 and c2.rg_gt_aux == {"enabled": True, "lambda": 0.25, "max_parents": 8}
    assert O.CKE2E2Config.from_any(c2.to_dict()) == c2
    with pytest.raises(ValueError, match="unknown"):
        O.CKE2E2Config.from_any({"v4": {"rg_gt_aux": {"lam": 1.0}}})
    for bad in ({"v4": {"rg_gt_aux": {"enabled": True}}},                      # mean readout
                {"v4": dict(v, rg_gt_aux={"enabled": True, "lambda": -1.0})},
                {"v4": dict(v, rg_gt_aux={"enabled": True, "max_parents": 0})},
                {"v4": dict(v, rg_read="bilinear")}):
        with pytest.raises(ValueError, match="rg_gt_aux|rg_read"):
            O.CKE2E2Config.from_any({**O.CKE2E2Config().to_dict(), **bad, "enabled": True,
                                     "io_dir": "/tmp/x"}).validate(None)
    # the launcher's set files declare an optional block at its default before the lines (launch_util2.apply_sets2)
    from tools.ck.e2e2 import launch_util2 as LU2
    ck = LU2.apply_sets2({"v4": dict(O._d_v4())}, ["v4.rg_gt_aux.enabled=true", "v4.rg_read=any4"])
    assert ck["v4"]["rg_gt_aux"]["enabled"] in (True, "true") and ck["v4"]["rg_read"] == "any4"


def _range_run(root: Path, arm: str, v4: V4Cfg, name: str) -> Path:
    src = TL.SMOKE_T if arm == "T" else TL.SMOKE_M
    rd = Path(root) / name
    rd.mkdir(parents=True, exist_ok=True)
    nf = "norm.npz" if arm == "T" else "norm_map.npz"
    shutil.copy(src / nf, rd / nf)
    torch.manual_seed(5)
    net = CKNet(arm, seed=1 if arm == "T" else 2, norm=load_arm_norm(rd, arm), v4=v4, corridor_version=2)
    with torch.no_grad():
        net.trunk.lon_head[-1].weight.zero_()
        net.trunk.lon_head[-1].bias.zero_()
    cfg = {"arm": arm, "seed": 1 if arm == "T" else 2, "trainer": "train_ck2", "v4": v4.to_dict(), "corridor_version": 2,
           "score_hidden": 256, "lead_aux": 0}
    save_ck(rd / "ckpt_last.pt", net, cfg)
    (rd / "config.json").write_text(json.dumps(cfg))
    (rd / "done.json").write_text("{}")
    return rd


V43S = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
        "pref_head": True, "corridor_version": 2, "kd_parent_only": True}


@have
def test_rg_gt_aux_in_the_student_loss(tmp_path):
    """warm-up step of a range student: rg_gt_aux on adds exactly lambda * (occ + margin) (same student, same KD) and
    trains the student's own field heads; tokens without GT -> no term; independent of kd_constraint (off here)."""
    out = {}
    for on in (False, True, "nogt"):
        v4 = dict(V43S, rg_read="any4", rg_stopped_aware=True,
                  rg_gt_aux={"enabled": bool(on), "lambda": 0.5, "max_parents": 8})
        c = TL.cfg2(tmp_path / f"on{on}", v4=v4, kd_score_keys=["nc", "dac"], lat_kd={"start_mb": 0})
        ck = O.CKE2E2(c)
        st = TLOSS._student(c)
        f, t = TL.fresh(2)
        t["ref_gt_ok"] = torch.tensor([False, False]) if on == "nogt" else torch.tensor([True, True])
        loss, logs = ck.loss(st, f, t, TL.fake_predictions(2, seed=0))
        loss.backward()
        out[on] = (float(loss), {k: float(v) for k, v in logs.items()}, st)
    (l0, g0, _), (l1, g1, st1), (l2, g2, _) = out[False], out[True], out["nogt"]
    assert "v4/rg_gt_occ" not in g0 and g1["v4/rg_n"] > 0 and np.isfinite(g1["v4/rg_gt_occ"])
    assert 0 < g1["v4/rg_rate_B"] <= g1["v4/rg_rate_U"] < 1 and g1["v4/rg_sdf_valid"] > 0
    assert 0 < g1["v4/rg_rate_Bp"] <= g1["v4/rg_rate_U"]                          # the Bp group's target is trained
    assert abs(g1["ck2/bce"] - g0["ck2/bce"]) < 1e-6 and abs(g1["ck2/kd_score"] - g0["ck2/kd_score"]) < 1e-6
    lw = TL.cfg2(tmp_path / "x").loss_weight
    assert abs((l1 - l0) - lw * 0.5 * (g1["v4/rg_gt_occ"] + g1["v4/rg_gt_margin"])) < 1e-4
    for head in (st1.v4_readout.blk_head, st1.v4_readout.marg_head):
        assert float(head[0].weight.grad.abs().sum()) > 0
    assert float(st1.v4_readout.blk_head[2].weight.grad[40:60].abs().sum()) > 0     # Bp rows of the last layer
    assert g2["v4/rg_n"] == 0.0 and g2["v4/rg_gt_occ"] == 0.0


@have
def test_constraint_teachers_b_semantics_check(tmp_path):
    """constraint KD: the DET teacher's B rule (rg_stopped_aware) must be the student's; rg_read may differ."""
    T_sa = _range_run(tmp_path, "T", V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True), "T_r2")
    T_pl = _range_run(tmp_path, "T", V4Cfg.range_arm(), "T_plain")
    M = _range_run(tmp_path, "M", V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True), "M_r2")
    v4 = dict(V43S, rg_read="nearest", rg_stopped_aware=True, kd_constraint={"enabled": True})
    ok = TL.cfg2(tmp_path / "io1", v4=v4, teacher_det_run=str(T_sa), teacher_map_run=str(M))
    O.CKE2E2(ok).teachers("cpu")                                            # same B rule, other read: accepted
    bad = TL.cfg2(tmp_path / "io2", v4=v4, teacher_det_run=str(T_pl), teacher_map_run=str(M))
    with pytest.raises(ValueError, match="rg_stopped_aware"):
        O.CKE2E2(bad).teachers("cpu")
