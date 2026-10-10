"""CK Phase 2 model hooks on the real v2 ParaSSRAgent (CPU): flag off = v2 bit-identical (vs the HEAD version of
para_ssr_agent.py), flag on = forward / compute_loss / backward with the real model and real targets (2 navtrain_train
tokens, fixture built once under ck/phase2/impl-model/fixtures), eval ck_* outputs, optimiser groups, checkpoint keys.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
FIX_DIR = Path("/workspace/yongjae/ssd/yongjae_refiner/ck/phase2/impl-model/fixtures")
FIX = FIX_DIR / "real_b2.pt"
ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
SCORES = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256"
FIX_TOKENS_ROWS = (0, 40000)            # packed navtrain_train rows used for the fixture

V2_OVERRIDES = dict(
    max_epochs=30, use_task_interaction=True, use_det_motion_head=True, use_map_head=True,
    grad_balance_target={"plan": 0.4, "det": 0.3, "map": 0.3}, plan_anchor=True, plan_anchor_file=ANCHORS,
    plan_score_file=SCORES, image_architecture="resnet34.tv_in1k", plan_heading_from_xy=True,
    backbone_pretrained=False,
)


def v2_config(**over):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    y = OmegaConf.load(REPO / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    c = OmegaConf.to_container(y.config, resolve=True)
    c.update(V2_OVERRIDES)
    c.update(over)
    return instantiate(c)


def sampling():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return TrajectorySampling(time_horizon=4.0, interval_length=0.5)


def ck_cfg(tmp: Path, **over):
    d = {"enabled": True, "io_dir": str(tmp / "ck_e2e")}
    d.update(over)
    return d


def build_agent(seed: int = 0, cls=None, **cfg_over):
    if cls is None:
        from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent as cls
    torch.manual_seed(seed)
    return cls(v2_config(**cfg_over), sampling(), lr=1e-4)


# ----------------------------------------------------------------------------------------------- real batch fixture
def make_fixture(path: Path = FIX) -> Path:
    """features + targets (all v2 builders + CKE2ETargetBuilder) of 2 packed navtrain_train tokens, collated."""
    import pandas as pd
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.common.dataloader import SceneLoader

    tdf = pd.read_parquet("/home/external-user/ssd/yongjae_refiner/ck/packed/navtrain_train/tokens.parquet")
    sub = tdf.iloc[list(FIX_TOKENS_ROWS)]
    toks, logs = [str(t) for t in sub.token], sorted(set(str(x) for x in sub.log))
    sp = Cn.SPLITS["navtrain_train"]
    agent = build_agent(0, ck_e2e=ck_cfg(path.parent / "unused"))
    sf = instantiate(OmegaConf.load(REPO / sp["scene_filter"]))
    sf = replace(sf, tokens=toks, max_scenes=None, log_names=logs)
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]),
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    feats, targs = [], []
    for tok in toks:
        scene = loader.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        f, t = {}, {}
        for b in agent.get_feature_builders():
            f.update(b.compute_features(ai))
        for b in agent.get_target_builders():
            t.update(b.compute_targets(scene))
        feats.append(f)
        targs.append(t)
    st = lambda xs: {k: torch.stack([x[k] for x in xs]) for k in xs[0]}  # noqa: E731
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save({"tokens": toks, "rows": list(FIX_TOKENS_ROWS), "features": st(feats), "targets": st(targs)}, tmp)
    os.replace(tmp, path)
    return path


@pytest.fixture(scope="module")
def batch():
    if not FIX.is_file():
        make_fixture(FIX)
    d = torch.load(FIX, map_location="cpu", weights_only=False)
    return d


def _eq(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _fresh(d):
    return ({k: v.clone() for k, v in d["features"].items()}, {k: v.clone() for k, v in d["targets"].items()})


def _v2_targets(t):
    drop = ("ck_", "ref_", "kd_bev_", "kd_ok_")
    return {k: v for k, v in t.items() if not k.startswith(drop)}


def head_agent_class():
    """ParaSSRAgent from the git HEAD version of para_ssr_agent.py (before the CK hooks)."""
    src = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:navsim/agents/para_ssr/para_ssr_agent.py"],
                         capture_output=True, text=True, check=True).stdout
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"_head_para_ssr_agent_{os.getpid()}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("navsim.agents.para_ssr._head_para_ssr_agent", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ParaSSRAgent


# ----------------------------------------------------------------------------------------------- flag off
def test_off_is_v2_bit_identical(batch):
    """ck_e2e {} (default) and {'enabled': False}: same state_dict keys / values, param groups, target builders,
    callbacks, forward outputs, loss and gradients as the HEAD agent."""
    Head = head_agent_class()
    ref = build_agent(0, cls=Head)
    agents = [build_agent(0), build_agent(0, ck_e2e={"enabled": False})]
    sd0 = ref.state_dict()
    for a in agents:
        assert a._ck_e2e is None and not hasattr(a, "ck_student")
        sd = a.state_dict()
        assert list(sd) == list(sd0)
        assert all(_eq(sd[k], sd0[k]) for k in sd0)
        g0, g1 = ref.get_optimizers()["optimizer"].param_groups, a.get_optimizers()["optimizer"].param_groups
        assert len(g0) == len(g1) == 2 and [len(g["params"]) for g in g0] == [len(g["params"]) for g in g1]
        assert [type(b).__name__ for b in ref.get_target_builders()] == [type(b).__name__ for b in
                                                                           a.get_target_builders()]
        assert [type(c).__name__ for c in ref.get_training_callbacks()] == [type(c).__name__ for c in
                                                                             a.get_training_callbacks()]
    # training forward + loss + grads (same RNG stream: GridMask / dropout)
    res = []
    for a in [ref] + agents:
        a.train()
        f, t = _fresh(batch)
        t = _v2_targets(t)
        torch.manual_seed(123)
        pred = a.forward(f)
        loss = a.compute_loss(f, t, pred)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in a.named_parameters() if p.grad is not None}
        res.append((pred, loss.detach(), {k: v.detach().clone() for k, v in a.latest_logs.items()}, grads))
    p0, l0, lg0, g0 = res[0]
    for p, l, lg, g in res[1:]:
        assert torch.equal(l, l0)
        assert set(p) == set(p0) and all(_eq(p[k], p0[k]) for k in p0)
        assert set(lg) == set(lg0) and all(_eq(lg[k], lg0[k]) for k in lg0)
        assert set(g) == set(g0) and all(torch.equal(g[k], g0[k]) for k in g0)
    # eval forward
    for a in [ref] + agents:
        a.eval()
    with torch.no_grad():
        outs = [a.forward(_fresh(batch)[0]) for a in [ref] + agents]
    for o in outs[1:]:
        assert set(o) == set(outs[0]) and all(_eq(o[k], outs[0][k]) for k in o)


# ----------------------------------------------------------------------------------------------- flag on
def test_on_v2_part_unchanged_and_ck_trains(batch, tmp_path):
    """enabled: v2 parameters / v2 loss identical to off (student built under fork_rng), CK loss added after the
    balancer, finite, every student parameter receives a gradient, CK optimiser group lr x3, ckpt keys."""
    off = build_agent(0)
    on = build_agent(0, ck_e2e=ck_cfg(tmp_path))
    sd_off, sd_on = off.state_dict(), on.state_dict()
    ck_keys = [k for k in sd_on if k.startswith("ck_student.")]
    assert ck_keys and [k for k in sd_on if not k.startswith("ck_student.")] == list(sd_off)
    assert all(_eq(sd_on[k], sd_off[k]) for k in sd_off)
    assert not any("teacher" in k for k in sd_on)
    assert on.ck_student.trunk.adapter.bev_grad_scale == 1.0
    # score prior from the Phase 1 labels
    from navsim.agents.para_ssr.ck.online import phase1_label_prior
    prior = phase1_label_prior("/home/external-user/ssd/yongjae_refiner/ck/labels/navtrain_train/cand")
    b = on.ck_student.score_head[-1].bias.detach().double().numpy()
    np.testing.assert_allclose(1 / (1 + np.exp(-b)), np.clip(prior, 1e-4, 1 - 1e-4), rtol=1e-5)
    # optimiser
    groups = on.get_optimizers()["optimizer"].param_groups
    assert len(groups) == 3 and groups[2]["lr_scale"] == 3.0 and groups[2]["weight_decay"] == 0.01
    assert abs(groups[2]["initial_lr"] - 3e-4) < 1e-12          # peak lr x3 (WarmupCosLR applies lr_scale)
    assert abs(groups[2]["lr"] - 3.0 * groups[0]["lr"]) < 1e-12
    assert {id(p) for p in groups[2]["params"]} == {id(p) for p in on.ck_student.parameters()}
    # builders / callbacks
    assert type(on.get_target_builders()[-1]).__name__ == "CKE2ETargetBuilder"
    assert type(on.get_training_callbacks()[-1]).__name__ == "CKE2ECallback"
    # train step (replay, epoch 0)
    res = []
    for a in (off, on):
        a.train()
        f, t = _fresh(batch)
        if a is off:
            t = _v2_targets(t)
        torch.manual_seed(7)
        pred = a.forward(f)
        loss = a.compute_loss(f, t, pred)
        loss.backward()
        res.append((loss.detach(), dict(a.latest_logs)))
    (l_off, lg_off), (l_on, lg_on) = res
    assert torch.equal(lg_on["loss_v2"], l_off)
    assert torch.isfinite(l_on) and float(lg_on["ck/loss"]) > 0
    assert abs(float(l_on) - float(l_off) - float(lg_on["ck/loss"])) < 1e-4 * max(1.0, float(l_on))
    for k in ("plan", "det", "map"):
        for key in lg_off:
            if key.startswith("loss_") and k in key and key in lg_on:
                assert torch.equal(lg_on[key], lg_off[key])
    assert float(lg_on["ck/phase"]) == 0.0 and float(lg_on["ck/G_src_phase1"]) == 1.0
    missing = [n for n, p in on.ck_student.named_parameters() if p.grad is None]
    assert not missing, missing
    gbev = [p.grad for n, p in on.para_ssr_model.named_parameters() if p.grad is not None]
    assert gbev and all(torch.isfinite(g).all() for g in gbev)
    for k, v in lg_on.items():
        assert torch.isfinite(v).all(), k


def test_on_eval_outputs(batch, tmp_path):
    on = build_agent(0, ck_e2e=ck_cfg(tmp_path)).eval()
    with torch.no_grad():
        pred = on.forward(_fresh(batch)[0])
    B = 2
    shapes = {"ck_cand": (B, 16, 8, 3), "ck_cand_idx": (B, 16), "ck_v2_final": (B, 16), "ck_v2_im": (B, 16),
              "ck_v2_sim": (B, 16, 5), "ck_score_logit": (B, 16, 5), "ck_z_lon": (B, 16, 6), "ck_w_lat": (B, 16, 6),
              "ck_c_lon": (B, 16, 6), "ck_e_lat": (B, 16, 6), "ck_corr_traj": (B, 16, 8, 3),
              "ck_corr_score_logit": (B, 16, 5)}
    for k, s in shapes.items():
        assert tuple(pred[k].shape) == s, k
        assert torch.isfinite(pred[k].float()).all(), k
    assert pred["ck_cand_idx"].dtype == torch.int64
    assert float((pred["ck_cand"][:, 0] - pred["trajectory"].float()).abs().max()) < 1e-4
    off = build_agent(0).eval()
    with torch.no_grad():
        p_off = off.forward(_fresh(batch)[0])
    assert torch.equal(p_off["trajectory"], pred["trajectory"])
    # zero-initialised control heads -> identity correction
    assert torch.allclose(pred["ck_corr_traj"], pred["ck_cand"], atol=1e-5)


def test_initialize_strict_roundtrip(batch, tmp_path):
    a = build_agent(0, ck_e2e=ck_cfg(tmp_path))
    with torch.no_grad():
        for p in a.ck_student.parameters():
            p.add_(0.01)
    ck = tmp_path / "x.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in a.state_dict().items()}}, ck)
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    b = ParaSSRAgent(v2_config(ck_e2e=ck_cfg(tmp_path)), sampling(), checkpoint_path=str(ck))
    b.initialize()
    for k, v in a.ck_student.state_dict().items():
        assert torch.equal(b.ck_student.state_dict()[k], v)
    # a v2-only checkpoint does not load into a CK agent (strict)
    off = build_agent(0)
    ck2 = tmp_path / "off.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in off.state_dict().items()}}, ck2)
    c = ParaSSRAgent(v2_config(ck_e2e=ck_cfg(tmp_path)), sampling(), checkpoint_path=str(ck2))
    with pytest.raises(RuntimeError, match="Missing key"):
        c.initialize()
