"""fusion_v1: aux_distill's v8 BEV distillation on the kyungmin-v2 core (CPU).

off: nothing is built, no scene_token, the target cache name is unchanged (the full bit-identity against ck2 is a
     cross-tree check recorded with the merge).
on (distill_selector_v8, the real teacher caches, the CK2 2-token fixture):
  - the v2 init and the global CPU RNG stream are those of the off run (probe drawn under fork_rng), the probe is
    frozen at construction (plain DDP), only anchor_distill.student_proj trains, in an optimiser group of its own;
  - one training step: loss = pre-distillation loss + distill_loss_weight * (bevfusion + resmap), gradients reach
    student_proj and the shared positional encoding, none reaches the frozen probe;
  - the recomputed bev_pos equals the planner's own;
  - teacher-store orientation against the refiner targets kd_bev_0 / kd_bev_1;
  - checkpoint round trips through initialize() in both directions, eval forces the distillation off, the
    training-set filter, the anchor-planner guard.
  cd /workspace/yongjae/SSR-fusion && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_fusion_v8_port.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_ck_e2e_model_agent import sampling, v2_config  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
ROOT = Path("/workspace/teacher_cache")
V8 = dict(use_distill=True, distill_selector_v8=True, distill_feature_root=str(ROOT), use_corridor_mask=False,
          distill_use_role_masks=False, distill_mse_weight=0.0)
TRAINABLE = "anchor_distill.student_proj."

pytestmark = pytest.mark.skipif(
    not (FIX2.is_file() and (ROOT / "bevfusion" / "cache_train_50x100").is_dir() and (ROOT / "resmap").is_dir()),
    reason="CK2 fixture / teacher caches absent")


def build(**over):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent

    torch.manual_seed(0)
    agent = ParaSSRAgent(v2_config(**over), sampling(), lr=1e-4)
    return agent, torch.get_rng_state()


def v2_targets(batch, scene_token: bool):
    drop = ("ck_", "ck2_", "ref_", "kd_bev_", "kd_ok_")
    t = {k: v.clone() for k, v in batch["targets"].items() if not k.startswith(drop)}
    if scene_token:
        t["scene_token"] = list(batch["tokens"])
    return t


def train_step(agent, batch, scene_token: bool, seed: int = 123):
    agent.train()
    f = {k: v.clone() for k, v in batch["features"].items()}
    t = v2_targets(batch, scene_token)
    torch.manual_seed(seed)
    pred = agent.forward(f)
    loss = agent.compute_loss(f, t, pred)
    loss.backward()
    return loss.detach(), dict(agent.latest_logs)


@pytest.fixture(scope="module")
def batch():
    return torch.load(FIX2, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def pair():
    off, rng_off = build()
    on, rng_on = build(**V8)
    return off, rng_off, on, rng_on


def _eq(a, b):
    return torch.equal(a, b) if torch.is_tensor(a) else a == b


# ----------------------------------------------------------------------------------------------- off
def test_off_builds_nothing_and_keeps_the_target_cache_name():
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder

    cfg = v2_config()
    assert cfg.use_distill is False and cfg.needs_scene_token is False
    knobs = v2_config(**dict(V8, use_distill=False))         # v8 knobs set, distillation off
    assert ParaSSRTargetBuilder(cfg, sampling()).get_unique_name() == \
        ParaSSRTargetBuilder(knobs, sampling()).get_unique_name()
    on = v2_config(**V8)
    assert on.needs_scene_token is True
    assert ParaSSRTargetBuilder(on, sampling()).get_unique_name() != \
        ParaSSRTargetBuilder(cfg, sampling()).get_unique_name()
    off, _ = build(**dict(V8, use_distill=False))
    assert getattr(off, "_distill", None) is None and off.distill_parameters() == []
    assert not any(k.startswith("_distill.") for k in off.state_dict())


# ----------------------------------------------------------------------------------------------- on: construction
def test_on_keeps_v2_init_and_rng_and_freezes_the_probe(pair):
    off, rng_off, on, rng_on = pair
    assert torch.equal(rng_off, rng_on)                      # global CPU RNG stream (data order) unchanged
    sd_off, sd_on = off.state_dict(), on.state_dict()
    extra = sorted(k for k in sd_on if k not in sd_off)
    assert extra and all(k.startswith("_distill.anchor_distill.") for k in extra)
    assert "_distill.anchor_distill.spatial_bias" not in sd_on   # rebuilt from plan_anchor_file, not saved
    assert all(_eq(sd_off[k], sd_on[k]) for k in sd_off)     # v2 init identical
    trainable = [n for n, p in on._distill.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(TRAINABLE) for n in trainable)
    assert len(trainable) == 4                               # 2 banks x (weight, bias)
    frozen = [n for n, p in on._distill.named_parameters() if not p.requires_grad]
    assert any(n.startswith("anchor_distill.q_proj.") for n in frozen)
    assert on._distill.anchor_distill.spatial_bias.shape == (4, 256, 5000)


def test_on_optimizer_group(pair):
    off, _, on, _ = pair
    g_off = off.get_optimizers()["optimizer"].param_groups
    g_on = on.get_optimizers()["optimizer"].param_groups
    assert len(g_on) == len(g_off) + 1
    assert [len(g["params"]) for g in g_on[:-1]] == [len(g["params"]) for g in g_off]
    last = g_on[-1]
    assert len(last["params"]) == 4 and last["lr_scale"] == 1.0 and last["weight_decay"] == g_on[0]["weight_decay"]
    ids = {id(p) for p in on.distill_parameters()}
    assert ids == {id(p) for p in last["params"]}


def test_anchor_planner_is_required():
    with pytest.raises(ValueError, match="anchor planner"):
        build(**dict(V8, plan_anchor=False))
    with pytest.raises(NotImplementedError, match="trajectory-anchor path"):
        build(**dict(V8, distill_selector_v8=False))


# ----------------------------------------------------------------------------------------------- on: one step
def test_on_training_step(pair, batch):
    off, _, on, _ = pair
    for a in (off, on):
        a.zero_grad(set_to_none=True)
    l_off, lg_off = train_step(off, batch, scene_token=False)
    l_on, lg_on = train_step(on, batch, scene_token=True)
    assert torch.equal(lg_on["loss_pre_distill"], l_off)    # the v2 part is the off run's loss
    w = float(on.config.distill_loss_weight)
    total = lg_on["loss_pre_distill"] + w * (lg_on["loss_distill_bevfusion"] + lg_on["loss_distill_resmap"])
    assert torch.isfinite(l_on) and torch.allclose(l_on, total, rtol=1e-6, atol=1e-5)
    for k in ("distill/effective_anchors", "distill_cos/bevfusion", "distill_cos/resmap", "gnorm/distill_v8"):
        assert k in lg_on and torch.isfinite(lg_on[k])
    for n, p in on._distill.named_parameters():
        if n.startswith(TRAINABLE) and n.endswith("weight"):
            assert p.grad is not None and float(p.grad.norm()) > 0
        if not p.requires_grad:
            assert p.grad is None
    pe_on = on.para_ssr_model.pts_bbox_head.positional_encoding
    pe_off = off.para_ssr_model.pts_bbox_head.positional_encoding
    g_on = torch.cat([p.grad.flatten() for p in pe_on.parameters()])
    g_off = torch.cat([p.grad.flatten() for p in pe_off.parameters()])
    assert not torch.equal(g_on, g_off)                      # the distillation reaches the shared positional encoding


def test_bev_pos_equals_the_planners(pair, batch):
    _, _, on, _ = pair
    head = on.para_ssr_model.pts_bbox_head
    seen = []
    handle = head.positional_encoding.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
    try:
        on.zero_grad(set_to_none=True)
        train_step(on, batch, scene_token=True, seed=7)
    finally:
        handle.remove()
    # encoder (3D mask), planner (plan_from_bev) and the distillation each call it; the planner's and ours agree
    planner_like = [o for o in seen if o.shape[-2:] == (head.bev_h, head.bev_w)]
    assert len(planner_like) >= 2 and torch.equal(planner_like[-1], planner_like[-2])


# ----------------------------------------------------------------------------------------------- teachers
def test_teacher_store_orientation_vs_refiner_targets(pair, batch):
    """BEVFusion: both pipelines flip W (equal).  ReSMap: the refiner pipeline transposes only (GT-validated);
    aux_distill's store flips W on top unless the branch sets flip_w false."""
    _, _, on, _ = pair
    toks = list(batch["tokens"])
    t = batch["targets"]
    bf = on._distill.stores["bevfusion"].load_batch(toks, torch.device("cpu"), torch.float32)
    assert torch.equal(bf, t["kd_bev_0"].float())
    from navsim.agents.para_ssr.distill.teacher_store import TeacherFeatureStore

    mirrored = TeacherFeatureStore(str(ROOT), "resmap", flip_w=True).load_batch(toks, torch.device("cpu"),
                                                                               torch.float32)
    straight = TeacherFeatureStore(str(ROOT), "resmap", flip_w=False).load_batch(toks, torch.device("cpu"),
                                                                                torch.float32)
    assert torch.equal(straight, t["kd_bev_1"].float())
    assert torch.equal(mirrored, t["kd_bev_1"].float().flip(-1))


# ----------------------------------------------------------------------------------------------- checkpoints / eval
def _save(agent, path: Path):
    torch.save({"state_dict": {"agent." + k: v for k, v in agent.state_dict().items()}}, path)
    return path


def test_checkpoint_round_trips(pair, tmp_path):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent

    off, _, on, _ = pair
    ck_on, ck_off = _save(on, tmp_path / "on.ckpt"), _save(off, tmp_path / "off.ckpt")
    # a distillation checkpoint evaluates with the distillation off ...
    a = ParaSSRAgent(v2_config(), sampling(), lr=1e-4, checkpoint_path=str(ck_on))
    a.initialize()
    assert all(_eq(a.state_dict()[k], v) for k, v in off.state_dict().items() if not k.startswith("_loss."))
    # ... and a distillation run can start from a checkpoint that never had it (its module keeps the init)
    b = ParaSSRAgent(v2_config(**V8), sampling(), lr=1e-4, checkpoint_path=str(ck_off))
    probe = {k: v.clone() for k, v in b._distill.state_dict().items()}
    b.initialize()
    assert all(torch.equal(b._distill.state_dict()[k], v) for k, v in probe.items())
    # on -> on restores the trainable student_proj too
    with torch.no_grad():
        for p in on.distill_parameters():
            p.add_(0.25)
    _save(on, ck_on)
    c = ParaSSRAgent(v2_config(**V8), sampling(), lr=1e-4, checkpoint_path=str(ck_on))
    c.initialize()
    assert all(torch.equal(x, y) for x, y in zip(c.distill_parameters(), on.distill_parameters()))


def test_eval_config_forces_the_distillation_off(tmp_path):
    from navsim.planning.script import run_aux_evaluation as runner

    cfg = tmp_path / "training.yaml"
    cfg.write_text("agent:\n  checkpoint_path: null\n  config:\n    use_distill: true\n    plan_anchor: true\n"
                   "    plan_score_file: /tmp/pdm_score_256\n", encoding="utf-8")
    loaded = runner._load_training_agent_config(cfg, tmp_path / "model.ckpt")
    assert loaded.config.use_distill is False


def test_filter_keeps_val_and_drops_uncached_train_tokens(pair):
    _, _, on, _ = pair

    class Toks:
        def __init__(self, tokens):
            self.tokens = list(tokens)

    good = "1aa44d46e4ab5bc7"
    train, val = Toks([good, "ffffffffffffffff"]), Toks(["ffffffffffffffff"])
    on.filter_datasets_to_teacher_cache(train, val)
    assert train.tokens == [good] and val.tokens == ["ffffffffffffffff"]


def test_default_resmap_store_is_transpose_only(pair, batch):
    """fusion_v1 default (distill_branches.resmap.flip_w False): the ReSMap store equals the refiner's
    GT-validated kd_bev_1; BEVFusion keeps its W flip."""
    _, _, on, _ = pair
    assert on._distill.stores["resmap"].flip_w is False and on._distill.stores["bevfusion"].flip_w is True
    rm = on._distill.stores["resmap"].load_batch(list(batch["tokens"]), torch.device("cpu"), torch.float32)
    assert torch.equal(rm, batch["targets"]["kd_bev_1"].float())


# ----------------------------------------------------------------------------------------------- fusion arm
def _ck2_cfg(tmp):
    from test_ck2_e2e_agent import ck2_cfg

    return ck2_cfg(tmp)


def _fusion_step(agent, batch, ok0=None, seed: int = 11):
    agent.train()
    agent.zero_grad(set_to_none=True)
    f = {k: v.clone() for k, v in batch["features"].items()}
    t = {k: v.clone() for k, v in batch["targets"].items()}
    t["scene_token"] = list(batch["tokens"])
    if ok0 is not None:
        t["kd_ok_0"] = torch.tensor(ok0)
    torch.manual_seed(seed)
    pred = agent.forward(f)
    loss = agent.compute_loss(f, t, pred)
    loss.backward()
    return loss.detach(), dict(agent.latest_logs)


SMOKE_T = D / "ck2/smoke/smoke_ck2T_ddp2"


@pytest.mark.skipif(not (SMOKE_T / "done.json").is_file(), reason="CK2 smoke teachers absent")
def test_targets_source_equals_store_source_on_the_ck2_arm(batch, tmp_path):
    """refiner e2e main arm + v8: teachers from the CK2 targets give exactly the store-fed loss (the default
    resmap orientation is the same transpose-only map) and gradients."""
    store, _ = build(ck_e2e2=_ck2_cfg(tmp_path / "s"), **V8)
    tgt, _ = build(ck_e2e2=_ck2_cfg(tmp_path / "t"), distill_teacher_source="targets", **V8)
    l_s, lg_s = _fusion_step(store, batch)
    l_t, lg_t = _fusion_step(tgt, batch)
    for k in ("loss_distill_bevfusion", "loss_distill_resmap", "distill/effective_anchors"):
        assert torch.equal(lg_s[k], lg_t[k]), k
    assert torch.equal(l_s, l_t)
    gs = dict(store.named_parameters())
    for n, p in tgt.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad, gs[n].grad), n
    assert float(lg_t["distill/teacher_ok_frac"]) == 1.0


@pytest.mark.skipif(not (SMOKE_T / "done.json").is_file(), reason="CK2 smoke teachers absent")
def test_targets_source_leaves_out_tokens_without_teacher(batch, tmp_path):
    """A token without teacher BEV contributes 0 and every kept token keeps its full-batch weight 1/B (ranks with
    different coverage weigh the same under DDP); with no teacher at all the term is 0 but the log keys are the same
    as in a full step (sync_dist logging) and student_proj still gets a (zero) gradient (plain DDP)."""
    agent, _ = build(ck_e2e2=_ck2_cfg(tmp_path / "a"), distill_teacher_source="targets", **V8)
    _, full = _fusion_step(agent, batch)
    _, lg = _fusion_step(agent, batch, ok0=[True, False])
    assert float(lg["distill/teacher_ok_frac"]) == 0.5 and torch.isfinite(lg["loss_distill_resmap"])
    added = lg["loss"] - lg["loss_pre_distill"]
    assert torch.allclose(added, 0.5 * (lg["loss_distill_bevfusion"] + lg["loss_distill_resmap"]), rtol=1e-5, atol=1e-5)
    _, lg = _fusion_step(agent, batch, ok0=[False, False])
    assert float(lg["distill/teacher_ok_frac"]) == 0.0
    dkeys = lambda d: {k for k in d if ("distill" in k or k.startswith("loss_attn")) and not k.startswith("gnorm/")}
    assert dkeys(lg) == dkeys(full) and len(dkeys(full)) >= 9    # the distillation keys (gnorm/* follow the cadence)
    assert float(lg["loss_distill_resmap"]) == 0.0 and torch.equal(lg["loss"], lg["loss_pre_distill"])
    for p in agent.distill_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) == 0.0


def test_spatial_bias_follows_the_planner_anchors(pair, batch):
    """The bias is rebuilt from the planner's anchor buffer when it differs from the anchors that built it (e.g. a
    checkpoint restore of a run whose anchor file changed); unchanged anchors leave it untouched."""
    _, _, on, _ = pair
    ad = on._distill.anchor_distill
    buf = on.para_ssr_model.pts_bbox_head.anchor_planner.trajectory_anchors
    assert torch.equal(ad.bias_anchors, buf)
    before, bias_id = ad.spatial_bias.clone(), ad.spatial_bias.data_ptr()
    assert ad.sync_spatial_bias(buf) is False and torch.equal(ad.spatial_bias, before)
    with torch.no_grad():
        buf[:, :, 1] += 1.0                                  # shift every anchor 1 m to the left
    try:
        on.zero_grad(set_to_none=True)
        train_step(on, batch, scene_token=True, seed=3)
        assert torch.equal(ad.bias_anchors, buf) and ad.spatial_bias.data_ptr() == bias_id    # rebuilt in place
        assert torch.equal(ad.spatial_bias, ad._build_spatial_bias(buf.clone()))
        assert not torch.equal(ad.spatial_bias, before)
    finally:
        with torch.no_grad():
            buf[:, :, 1] -= 1.0
        ad.sync_spatial_bias(buf)


def test_targets_source_needs_the_ck2_arm():
    with pytest.raises(ValueError, match="needs ck_e2e2"):
        build(distill_teacher_source="targets", **V8)
    with pytest.raises(ValueError, match="distill_teacher_source"):
        build(distill_teacher_source="disk", **V8)
