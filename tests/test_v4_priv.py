"""refiner v4 PRIVILEGED range-teacher input (V4Cfg.rg_priv; config docstring): the constant-velocity GT-object channels
(range_targets.priv_channels), their region rule, the readout term (zero-initialised: the fields are bitwise those of
the plain R2 net at init), the CKNet plumbing (priv= required / refused) and load_ck's privileged guard.  CPU."""
import json
from pathlib import Path

import pytest
import torch

from navsim.agents.para_ssr.ck.model import CKNet, load_ck, save_ck
from navsim.agents.para_ssr.ck.v4 import range_targets as RT
from navsim.agents.para_ssr.ck.v4.config import V4Cfg
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry
from navsim.agents.para_ssr.refiner.e2e import ego_inputs
from tests.test_v4_r2 import R2, _draft, _ref_scene, _status

PRIV = {r: V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True, rg_priv=r) for r in ("far", "out", "all")}


def _frame(cand):
    geom = corridor_geometry(cand.reshape(-1, 8, 3).float(), version=2)
    return RT.parent_frame(geom)


def _scene():
    """4 objects (x, y, h, vx, vy, L, agent): ahead in the window (stopped), far band (lead at 8 m/s), behind
    (closing at 14 m/s), far side."""
    return _ref_scene([(20.0, 0.0, 0.0, 0.0, 0.0, 4.5, True), (40.0, 0.0, 0.0, 8.0, 0.0, 4.5, True),
                       (-12.0, 0.0, 0.0, 14.0, 0.0, 4.5, True), (10.0, 40.0, 0.0, 0.0, 0.0, 4.5, True)])


def test_region_rule():
    x = torch.tensor([20.0, 40.0, -12.0, 10.0, 70.0])
    y = torch.tensor([0.0, 0.0, 0.0, 40.0, 0.0])
    assert RT.priv_region(x, y, "all").tolist() == [True] * 5
    assert RT.priv_region(x, y, "out").tolist() == [False, True, True, True, True]
    assert RT.priv_region(x, y, "far").tolist() == [False, True, False, False, False]
    with pytest.raises(ValueError):
        RT.priv_region(x, y, "off")


def test_priv_channels_regions_and_cv():
    """a straight 8 m/s draft: 'far' sees only the lead at 40 m (first bins at the far stations), 'out' adds the
    closing rear car (reaches the ego path later: its cells appear at later bins near the ego), 'all' adds the stopped
    car at 20 m from bin 0; tokens without GT give zeros; the row -> token map is honoured."""
    ref = _scene()
    cand = _draft(8.0)[None]                                          # [1, 8, 3]
    fr = _frame(cand)
    tok = torch.tensor([0])
    M, tb = 20, 0.25
    far = RT.priv_channels(ref, tok, fr, M, tb, "far")
    out = RT.priv_channels(ref, tok, fr, M, tb, "out")
    al = RT.priv_channels(ref, tok, fr, M, tb, "all")
    assert far.shape == (1, 48, 17, M) and far.dtype == torch.float32
    assert far.sum() > 0 and out.sum() >= far.sum() and al.sum() > out.sum()
    assert ((far > 0) & ~(out > 0)).sum() == 0 and ((out > 0) & ~(al > 0)).sum() == 0
    # the stopped car at 20 m (in the window): only 'all' has an occupied cell at bin 0 within 24 m
    s = (torch.arange(48) + 0.5)[:, None, None] * float(fr["ds"][0])
    near0 = (s < 24.0).expand(48, 17, M).clone()
    near0[..., 1:] = False
    assert al[0][near0].sum() > 0 and out[0][near0].sum() == 0 and far[0][near0].sum() == 0
    # the rear car closes in: 'out' has occupied cells at later bins that 'far' lacks
    assert ((out[0, ..., 8:] > 0) & ~(far[0, ..., 8:] > 0)).sum() > 0
    # no GT -> zeros; two rows of the same token -> identical channels
    ref0 = dict(ref, ref_obj_n=torch.tensor([0], dtype=torch.int32))     # no objects -> zeros
    assert RT.priv_channels(ref0, tok, fr, M, tb, "all").sum() == 0
    ref0 = dict(ref, ref_obj_ok=torch.tensor([False]))                    # loader flag off -> zeros
    assert RT.priv_channels(ref0, tok, fr, M, tb, "all").sum() == 0
    fr2 = _frame(torch.cat([cand, cand], 0))
    two = RT.priv_channels(ref, torch.tensor([0, 0]), fr2, M, tb, "all")
    assert torch.equal(two[0], two[1]) and torch.equal(two[0], al[0])
    with pytest.raises(ValueError):
        RT.priv_channels(ref, torch.tensor([0, 0, 0]), fr2, M, tb, "all")


def test_net_init_bitwise_and_plumbing():
    """at init the privileged net's fields / scores equal the plain R2 net's (zero-initialised term, same seeded
    modules); priv= is required (ValueError) and refused on a plain net; the extra set (variants) takes it too."""
    torch.manual_seed(0)
    n0 = CKNet("S", seed=0, v4=R2).eval()
    n1 = CKNet("S", seed=0, v4=PRIV["out"]).eval()
    sd0, sd1 = n0.state_dict(), n1.state_dict()
    extra = sorted(set(sd1) - set(sd0))
    assert extra == ["v4_readout.priv_blk_w", "v4_readout.priv_marg_w"]
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    ref = _scene()
    cand = torch.stack([_draft(8.0), _draft(6.0, 0.02)])[None]
    status = _status(8.0)
    bev = torch.randn(1, 256, 50, 100)
    with torch.no_grad():
        o0 = n0(bev, cand, status, decode=False)
        o1 = n1(bev, cand, status, decode=False, priv=ref)
    assert torch.equal(o0["score_logit"], o1["score_logit"])
    assert torch.equal(o0["haz"]["occ_logit_u"], o1["haz"]["occ_logit_u"])
    with pytest.raises(ValueError):
        n1(bev, cand, status, decode=False)
    with pytest.raises(ValueError):
        n0(bev, cand, status, decode=False, priv=ref)
    # a non-zero term changes the fields; the variants' set gets priv too (no error)
    with torch.no_grad():
        n1.v4_readout.priv_blk_w.fill_(0.5)
        o2 = n1(bev, cand, status, extra=cand, decode=False, priv=ref)
    assert not torch.equal(o0["haz"]["occ_logit_u"], o2["haz"]["occ_logit_u"])
    assert o2["extra_score_logit"].shape == o2["score_logit"].shape


def test_cfg_roundtrip_and_load_guard(tmp_path: Path):
    d = PRIV["far"].to_dict()
    assert d["rg_priv"] == "far" and V4Cfg.from_dict(d).rg_priv == "far"
    assert V4Cfg.from_dict(R2.to_dict()).rg_priv == "off"
    with pytest.raises(ValueError):
        V4Cfg.range_arm(rg_priv="rear")
    with pytest.raises(ValueError):                   # range-only field
        V4Cfg(readout="hazard", rg_priv="all").validate()
    net = CKNet("T", seed=0, v4=PRIV["far"])
    run = tmp_path / "v4Tx_rngPfar_PRIV"
    run.mkdir()
    cfg = {"arm": "T", "seed": 0, "corridor_version": 2, "v4": PRIV["far"].to_dict(), "privileged": 1}
    (run / "config.json").write_text(json.dumps(cfg))
    save_ck(run / "ckpt_last.pt", net, cfg)
    with pytest.raises(PermissionError):
        load_ck(run, "last")
    net2, _ = load_ck(run, "last", allow_privileged=True)
    assert net2.v4.rg_priv == "far" and net2.v4_readout.priv_blk_w is not None
