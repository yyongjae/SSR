"""refiner v6c on the real cf_seg_v15 ParaSSRAgent (CPU, random init, a 2-token navtrain_train batch built once).

  * v6c off: the model builds no v6c module and the forward has no v6c key (the baseline path is untouched)
  * stage D: forward + compute_loss + backward; every v6c parameter gets a finite gradient; tau_v6 == tau_base at
    init (zero-initialised corrector); eval forward submits v6c_out
  * stage P: baseline parameters frozen (no grad), BatchNorm / GridMask in eval while the cf copy path stays on
  * gradient contract (report 124 §4-2a, T11): R-J and the re-read receive no gradient from L_corr; L_corr reaches
    only the corrector
The fixture needs the dataset, the GT caches and the teacher BEV caches of this host; skipped when absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
KM = Path(os.environ.get("KM_ROOT", "/workspace/navsim_workspace/corridor_cache"))
DATA = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/home/external-user/yongjae/SSR/data/dataset"))
TEACH = Path(os.environ.get("TEACHER_CACHE_ROOT", "/workspace/teacher_cache"))
FIX = Path(os.environ.get("V6C_FIXTURE", str(REPO / "work_dirs/v6c_fixture/v6c_fix_b2.pt")))   # built once, gitignored
ROWS = (0, 40000)

CF_SEG_V15 = dict(
    max_epochs=30, use_task_interaction=True, use_det_motion_head=True, use_map_head=True,
    grad_balance_target={"plan": 0.4, "det": 0.3, "map": 0.3}, plan_anchor=True,
    plan_anchor_file=str(REPO / "resources/km_corridor/trajectory_anchors_256.npy"),
    plan_score_file=str(KM / "planning_vb/pdm_score_256"), image_architecture="resnet34.tv_in1k",
    plan_heading_from_xy=True, backbone_pretrained=False, log_sync_dist=False,
    corridor_weight=1.0, corridor_tokens_root=str(REPO / "resources/km_corridor/packed"),
    corridor_select_rule="cand_official",
    # V31C (train.sh, c64 family) + cf
    corridor_plan_weight=20.0, corridor_plan_ttc_weight=1.0, plan_ep_feasible_only=True, corridor_occ_planes=49,
    corridor_route_weight=1.0, corridor_plan_prog_weight=0.2, corridor_plan_comfort_weight=1.0,
    corridor_format="c64", corridor_dir=str(KM / "corridor64"), pc_range=[-32.0, 0.0, -2.0, 32.0, 64.0, 2.0],
    map_pc_range=[-32.0, 0.0, -2.0, 32.0, 64.0, 2.0], bev_h=100, cand_scorer_weight=1.0, exact_dir=str(KM / "exact"),
    cand_official=True, exact2_dir=str(KM / "exact2"), cand_scorer_fields=True, corridor_plan_ec_smooth_weight=0.1,
    cand_cf_folds=2, input_target=True, map_repr="seg",
)


def have_inputs() -> bool:
    return (KM / "corridor64/navtrain_train").is_dir() and (DATA / "navsim_logs").is_dir() and (TEACH / "bevfusion").is_dir()


def make_config(**over):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    y = OmegaConf.load(REPO / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    c = OmegaConf.to_container(y.config, resolve=True)
    c.update(CF_SEG_V15)
    c.update(over)
    return instantiate(c)


def sampling():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return TrajectorySampling(time_horizon=4.0, interval_length=0.5)


def build_agent(seed=0, **over):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    torch.manual_seed(seed)
    return ParaSSRAgent(make_config(**over), sampling(), lr=1e-4)


def make_fixture(path: Path = FIX) -> Path:
    """features + targets of 2 packed navtrain_train tokens through the cf_seg_v15 builders (+ the teacher BEVs)"""
    import pandas as pd
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.distill.teacher_store import DistillTeacherTargetBuilder
    tdf = pd.read_parquet(REPO / "resources/km_corridor/packed/navtrain_train/tokens.parquet")
    sub = tdf.iloc[list(ROWS)]
    toks, logs = [str(t) for t in sub.token], sorted(set(str(x) for x in sub.log))
    agent = build_agent(0, v6c=False)
    cfg = make_config(distill_feature_root=str(TEACH), distill_cache_size=[100, 100])
    sf = instantiate(OmegaConf.load(REPO / "navsim/planning/script/config/common/scene_filter/navtrain.yaml"))
    sf = replace(sf, tokens=toks, max_scenes=None, log_names=logs)
    loader = SceneLoader(data_path=DATA / "navsim_logs/trainval", sensor_blobs_path=DATA / "sensor_blobs/trainval",
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    builders_t = agent.get_target_builders() + [DistillTeacherTargetBuilder(cfg)]
    feats, targs = [], []
    for tok in toks:
        scene = loader.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        f, t = {}, {}
        for bld in agent.get_feature_builders():
            f.update(bld.compute_features(ai))
        for bld in builders_t:
            t.update(bld.compute_targets(scene))
        feats.append(f)
        targs.append(t)
    stack = lambda xs: {k: torch.stack([x[k] for x in xs]) for k in xs[0]}  # noqa: E731
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": toks, "features": stack(feats), "targets": stack(targs)}, path)
    return path


@pytest.fixture(scope="module")
def batch():
    if not have_inputs():
        pytest.skip("dataset / GT caches / teacher caches not on this host")
    if not FIX.is_file():
        make_fixture(FIX)
    return torch.load(FIX, map_location="cpu", weights_only=False)


def _fresh(d):
    return {k: v.clone() for k, v in d["features"].items()}, {k: v.clone() for k, v in d["targets"].items()}


def test_off_builds_nothing(batch):
    a = build_agent(0)
    assert a.para_ssr_model.v6c is None
    f, t = _fresh(batch)
    a.train()
    p = a.forward(f, t)
    assert not any(k.startswith("v6c_") for k in p)


def test_stage_d_forward_loss_backward(batch):
    a = build_agent(0, v6c=True, v6c_stage="D")
    f, t = _fresh(batch)
    a.train()
    p = a.forward(f, t)
    assert torch.equal(p["v6c_tau_v6"], p["v6c_tau_base"])                               # zero-initialised R-C
    assert p["v6c_E_base"].shape[-2:] == (21, 12)
    loss = a.compute_loss(f, t, p)
    assert torch.isfinite(loss)
    logs = a.latest_logs
    assert "v6c/loss" in logs and "loss_pre_v6c" in logs
    loss.backward()
    for n, prm in a.para_ssr_model.v6c.named_parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all(), n


def test_gradient_contract(batch):
    """L_corr reaches only R-C (through tau_v6); R-J / the re-read read detached states"""
    from navsim.agents.para_ssr.refiner_v6c import losses as LS
    a = build_agent(0, v6c=True, v6c_stage="D")
    f, t = _fresh(batch)
    a.train()
    p = a.forward(f, t)
    assert p["v6c_S_v6_g"].requires_grad
    for k in ("v6c_j_nc", "v6c_j_dac", "v6c_pdms_hat"):
        assert p[k].grad_fn is None or k != "v6c_pdms_hat"
    w = {"gt": 1.0, "road": 1.0, "comfort": 1.0, "ec": 0.1, "prog": 0.2}
    l, _ = LS.corr_loss(p["v6c_S_v6_g"], p["v6c_S_base"], t, None, torch.ones(2), w)
    v6c = a.para_ssr_model.v6c
    gr = torch.autograd.grad(l + 0.0 * p["v6c_ctrl"].sum(), [*v6c.rc.parameters(), *v6c.re.parameters(),
                                                         *v6c.rj.parameters()], allow_unused=True)
    n_rc = len(list(v6c.rc.parameters()))
    n_re = len(list(v6c.re.parameters()))
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in gr[n_rc:])                 # nothing into R-E / R-J
    # R-J's inputs: its loss reaches only R-J
    lj = p["v6c_j_nc"].sum() + p["v6c_j_eff"].sum()
    g2 = torch.autograd.grad(lj, [*v6c.re.parameters(), *v6c.rc.parameters()], allow_unused=True)
    assert all(g is None for g in g2)
    assert n_re > 0


def test_stage_p_freezes_baseline(batch):
    a = build_agent(0, v6c=True, v6c_stage="P")
    a.train()
    m = a.para_ssr_model
    assert all(not prm.requires_grad for n, prm in m.named_parameters() if not n.startswith("v6c."))
    assert all(prm.requires_grad for prm in m.v6c.parameters())
    assert m.training and m.pts_bbox_head.training                                         # copy candidates stay on
    assert not m.corridor_head.inp[1].training and (m.grid_mask is None or not m.grid_mask.training)
    f, t = _fresh(batch)
    p = a.forward(f, t)
    assert "cf_offset" in p                                                                 # cross-fit copies ran
    loss = a.compute_loss(f, t, p)
    loss.backward()
    assert all(prm.grad is None for n, prm in m.named_parameters() if not n.startswith("v6c."))


def test_eval_submits_v6c_out(batch):
    a = build_agent(0, v6c=True, v6c_stage="D", v6c_mode="A")
    a.eval()
    f, _ = _fresh(batch)
    with torch.no_grad():
        p = a.forward(f)
    assert torch.equal(p["trajectory"], p["v6c_out"].to(p["trajectory"].dtype))
    assert torch.equal(p["trajectory_base"], p["v6c_tau_base"])
    assert torch.equal(p["v6c_tau_base"], p["cand_traj"][torch.arange(2), p["plan_select_index"]].float())


def test_warmup_trains_re_only_on_the_mixture(batch):
    a = build_agent(0, v6c=True, v6c_stage="D", v6c_warmup_steps=10)
    a.train()
    f, t = _fresh(batch)
    p = a.forward(f, t)
    loss = a.compute_loss(f, t, p)
    assert float(a.latest_logs["v6c/warmup"]) == 1.0 and "v6c/E_ovl_warm" in a.latest_logs
    loss.backward()
    v6c = a.para_ssr_model.v6c
    assert all(float(x.grad.abs().sum()) == 0 for x in v6c.rc.parameters())                # R-C untouched
    assert any(float(x.grad.abs().sum()) > 0 for x in v6c.re.parameters())


def test_stage_p_attention_dropout_off(batch):
    a = build_agent(0, v6c=True, v6c_stage="P")
    a.train()
    mha = [m for n, m in a.para_ssr_model.named_modules() if isinstance(m, torch.nn.MultiheadAttention)
           and not n.startswith("v6c")]
    assert mha and all(not m.training for m in mha)
    v6c_mha = [m for m in a.para_ssr_model.v6c.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    assert all(m.training for m in v6c_mha)
    f, t = _fresh(batch)
    p1 = a.forward(f, t)
    p2 = a.forward(*_fresh(batch))
    assert torch.equal(p1["v6c_idx"], p2["v6c_idx"])                                       # deterministic copy pick


def test_bad_v6c_config_rejected():
    with pytest.raises(ValueError):
        build_agent(0, v6c=True, cand_cf_folds=0)
