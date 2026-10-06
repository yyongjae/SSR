"""Planning readout and readout-space distillation (report/19)."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_loss import ParaSSRLoss
from navsim.agents.para_ssr.readout.bev_cache import (
    BevCache,
    pseudo_map_targets,
    teacher_vectors_to_student,
)
from navsim.agents.para_ssr.readout.distill import ReadoutDistiller, kd_ramp
from navsim.agents.para_ssr.readout.readout import (
    build_readout,
    load_readout,
    metric_cell_centres,
    readout_checkpoint,
)
from navsim.agents.para_ssr.readout.teacher_targets import ResMapTeacherTargetBuilder

H, W, C = 50, 100, 256


def _batch(bs=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    cmd = torch.eye(4)[torch.arange(bs) % 4]
    return torch.randn(bs, C, H, W, generator=g), cmd, torch.randn(bs, 4, generator=g)


# ---------------------------------------------------------------------- #
# readout
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["h0", "h1", "h2"])
def test_readout_shapes_and_capacity_order(preset):
    model = build_readout(preset)
    bev, cmd, ego = _batch()
    out = model(bev, cmd, ego)
    assert out["z"].shape == (3, 1, 256)
    assert out["attn"].shape == (3, 1, H * W)
    assert out["ego_fut_preds"].shape == (3, 4, 8, 3)
    torch.testing.assert_close(out["trajectory"], out["ego_fut_preds"][:, 0].cumsum(1))


def test_capacity_increases_with_preset():
    n = [sum(p.numel() for p in build_readout(k).parameters()) for k in ("h0", "h1", "h2")]
    assert n[0] < n[1] < n[2]


def test_late_ego_keeps_z_ego_free_and_early_does_not():
    bev, cmd, ego = _batch()
    late = build_readout("h1", ego_inject="late").eval()
    torch.testing.assert_close(late.encode(bev, cmd, ego), late.encode(bev, cmd, ego + 5))
    early = build_readout("h1", ego_inject="early").eval()
    assert not torch.allclose(early.encode(bev, cmd, ego), early.encode(bev, cmd, ego + 5))


def test_ego_floor_ignores_the_bev():
    model = build_readout("h1", use_bev=False).eval()
    bev, cmd, ego = _batch()
    a = model(bev, cmd, ego)["trajectory"]
    b = model(None, cmd, ego)["trajectory"]
    torch.testing.assert_close(a, b)


def test_metric_cells_follow_student_layout():
    xy = metric_cell_centres((-32.0, 0.0, -2.0, 32.0, 32.0, 2.0), H, W).view(H, W, 2)
    torch.testing.assert_close(xy[0, 0], torch.tensor([-31.68, 0.32]))    # row 0 near, col 0 left
    torch.testing.assert_close(xy[-1, -1], torch.tensor([31.68, 31.68]))


def test_checkpoint_roundtrip(tmp_path):
    model = build_readout("h0", num_queries=2, cmd_inject="late")
    torch.save(readout_checkpoint(model), tmp_path / "r.pt")
    again = load_readout(tmp_path / "r.pt")
    assert again.cfg == model.cfg
    bev, cmd, ego = _batch()
    torch.testing.assert_close(again(bev, cmd, ego)["trajectory"], model(bev, cmd, ego)["trajectory"])


# ---------------------------------------------------------------------- #
# cache reader
# ---------------------------------------------------------------------- #
def _write_cache(root: Path, layout=None, frames=2):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    (root / "bev").mkdir()
    (root / "vectors").mkdir()
    (root / "scores").mkdir()
    (root / "labels").mkdir()
    shape = (C, W, H) if layout != "forward_right" else (C, H, W)
    bev = rng.standard_normal((frames, *shape)).astype(np.float16)
    np.save(root / "bev" / "r0_s0000.npy", bev)
    vec = np.zeros((frames, 100, 20, 2), np.float16)
    vec[:, 0, :, 0] = np.linspace(0.0, 1.0, 20)   # u: forward 0 -> 32 m
    vec[:, 0, :, 1] = 0.25                          # v: y_left = -16 m -> x_right = +16 m
    np.save(root / "vectors" / "r0_s0000.npy", vec)
    scores = np.zeros((frames, 100), np.float16)
    scores[:, 0] = 0.9
    np.save(root / "scores" / "r0_s0000.npy", scores)
    labels = np.full((frames, 100), 2, np.int8)
    np.save(root / "labels" / "r0_s0000.npy", labels)
    (root / "index.json").write_text(json.dumps({f"tok{i}": ["r0_s0000", i] for i in range(frames)}))
    meta = {"checkpoint_sha256": "abc"}
    if layout:
        meta["layout"] = layout
    (root / "meta.json").write_text(json.dumps(meta))
    return bev


def test_teacher_cache_is_transposed_student_cache_is_not(tmp_path):
    raw_t = _write_cache(tmp_path / "t")
    cache = BevCache(tmp_path / "t")
    assert cache.bev("tok1").shape == (C, H, W)
    np.testing.assert_array_equal(cache.bev("tok1")[:, 7, 3], raw_t[1, :, 3, 7])
    raw_s = _write_cache(tmp_path / "s", layout="forward_right")
    np.testing.assert_array_equal(BevCache(tmp_path / "s").bev("tok0"), raw_s[0])


def test_teacher_vector_convention():
    # u = x_forward / 32, v = (y_left + 32) / 64  ->  (x_right, y_forward) normalised
    out = teacher_vectors_to_student(np.array([[0.5, 1.0], [0.0, 0.25]]))
    np.testing.assert_allclose(out, [[0.0, 0.5], [0.75, 0.0]])


def test_pseudo_map_targets_match_gt_format(tmp_path):
    _write_cache(tmp_path / "t")
    cfg = ParaSSRConfig()
    t = pseudo_map_targets(
        BevCache(tmp_path / "t"), "tok0", score_thr=0.3, max_vec=cfg.map_max_vec,
        num_orders=cfg.map_num_orders, num_pts=cfg.map_num_pts_per_vec, pc_range=cfg.pc_range,
    )
    assert t["gt_map_pts"].shape == (100, 20, 20, 2)
    assert t["gt_map_valid"].sum() == 1 and t["gt_map_labels"][0] == 2
    first = t["gt_map_pts"][0, 0]
    np.testing.assert_allclose(first[:, 0], 0.75, atol=1e-6)            # x_right = +16 m
    np.testing.assert_allclose(first[[0, -1], 1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(t["gt_map_pts"][0, 1], first[::-1])      # reversed order


# ---------------------------------------------------------------------- #
# distillation
# ---------------------------------------------------------------------- #
def test_kd_ramp():
    assert kd_ramp(4, 5, 10) == 0.0
    assert kd_ramp(5, 5, 10) == pytest.approx(0.1)
    assert kd_ramp(100, 5, 10) == 1.0
    assert kd_ramp(0, 0, 0) == 1.0


def _kd_config(tmp_path, **kw):
    ckpt = tmp_path / "readout.pt"
    torch.save(readout_checkpoint(build_readout("h0")), ckpt)
    base = dict(
        kd_mode="readout", kd_readout_ckpt=str(ckpt), kd_distance="cosine", kd_weight=1.0,
        kd_warmup_iters=0, kd_ramp_iters=0, kd_adapter=False, kd_random_seed=0, bev_h=H, bev_w=W, embed_dims=C, pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _kd_inputs(bs=2):
    f_t, cmd, _ = _batch(bs, seed=1)
    bev_embed = torch.randn(bs, H * W, C, requires_grad=True)
    targets = {"teacher_bev": f_t.half(), "teacher_valid": torch.ones(bs), "command": cmd}
    return bev_embed, targets


def test_reader_is_frozen_and_not_a_submodule(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path))
    assert list(d.parameters()) == []
    assert "_reader" not in dict(d.named_modules())
    bev_embed, targets = _kd_inputs()
    d(bev_embed, targets, 0)["loss"].backward()
    assert bev_embed.grad is not None and bev_embed.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.__dict__["_reader"].parameters())


def test_distill_is_zero_when_student_matches_teacher(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path))
    _, targets = _kd_inputs()
    same = targets["teacher_bev"].float().permute(0, 2, 3, 1).reshape(2, H * W, C)
    assert d(same, targets, 0)["raw"] == pytest.approx(0.0, abs=1e-5)


def test_distill_masks_frames_missing_from_the_cache(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path))
    bev_embed, targets = _kd_inputs()
    targets["teacher_valid"] = torch.zeros(2)
    out = d(bev_embed, targets, 0)
    assert float(out["loss"]) == 0.0


def test_schedule_counts_from_the_first_distillation_step(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_warmup_iters=10))
    bev_embed, targets = _kd_inputs()
    assert float(d(bev_embed, targets, 5000)["coef"]) == 0.0   # a fine-tune's restored counter
    assert float(d(bev_embed, targets, 5010)["coef"]) == 1.0
    assert d.start_iter == 5000


@pytest.mark.parametrize("mode", ["random", "feature"])
def test_control_modes(tmp_path, mode):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode=mode, kd_distance="mse"))
    bev_embed, targets = _kd_inputs()
    out = d(bev_embed, targets, 0)
    out["loss"].backward()
    assert float(out["raw"]) > 1e-3 and bev_embed.grad.abs().sum() > 0


def test_adapter_starts_as_identity(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_adapter=True))
    x = torch.randn(1, H * W, C)
    torch.testing.assert_close(d.student_bev(x), x.view(1, H, W, C).permute(0, 3, 1, 2))


class _NoHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.det_motion_head = None
        self.map_head = None
        self.aux_grad_scale = {}


def test_loss_balances_distill_like_any_task(tmp_path):
    cfg = _kd_config(
        tmp_path,
        grad_balance_target={"plan": 0.5, "distill": 0.5},
        grad_balance_interval=1, grad_balance_momentum=0.0, grad_balance_clamp=(1e-5, 1.0),
        grad_balance_warmup_iters=0, grad_norm_log_interval=0,
        task_loss_weight={"plan": 1.0}, heading_weight=0.5,
        use_det_motion_head=False, use_map_head=False,
    )
    loss = ParaSSRLoss(cfg)
    assert set(loss.balancer.target) == {"plan", "distill"}
    model = _NoHeadModel().train()
    bev_embed, targets = _kd_inputs()
    targets.update(trajectory_offsets=torch.zeros(2, 8, 3), trajectory_mask=torch.ones(2, 8))
    preds = {
        "bev_embed": bev_embed,
        "ego_fut_preds": bev_embed[:, :24, 0].reshape(2, 1, 8, 3).expand(2, 4, 8, 3),
    }
    for _ in range(2):  # the balancer solves from the second iteration on
        total, logs = loss(model, {}, targets, preds)
        total.backward(retain_graph=True)
    assert "kd/raw" in logs and "gscale/distill" in logs
    assert 0 < loss.balancer.scale_for("distill") <= 1.0
    state = loss.get_extra_state()
    assert state["kd_start_iter"] == 0


def test_distill_target_dropped_without_kd():
    cfg = SimpleNamespace(
        kd_mode="none", grad_balance_target={"plan": 0.5, "distill": 0.5}, grad_balance_interval=1,
        grad_balance_momentum=0.9, grad_balance_clamp=(1e-5, 1.0), grad_balance_warmup_iters=0,
        use_det_motion_head=False, use_map_head=False,
    )
    assert ParaSSRLoss(cfg).balancer is None


def test_pseudo_labels_replace_gt_only_where_the_teacher_has_the_frame():
    cfg = SimpleNamespace(map_label_source="teacher")
    loss = ParaSSRLoss.__new__(ParaSSRLoss)
    loss._config = cfg
    targets = {
        "gt_map_pts": torch.zeros(2, 3, 4, 5, 2),
        "gt_map_labels": torch.zeros(2, 3, dtype=torch.long),
        "gt_map_valid": torch.zeros(2, 3, dtype=torch.bool),
        "teacher_map_pts": torch.ones(2, 3, 4, 5, 2),
        "teacher_map_labels": torch.ones(2, 3, dtype=torch.long),
        "teacher_map_valid": torch.ones(2, 3, dtype=torch.bool),
        "teacher_valid": torch.tensor([1.0, 0.0]),
    }
    logs = {}
    pts, labels, valid = loss._map_labels(targets, logs)
    assert pts[0].min() == 1 and pts[1].max() == 0
    assert labels.tolist() == [[1, 1, 1], [0, 0, 0]]
    assert valid.tolist() == [[True] * 3, [False] * 3]
    assert float(logs["kd/pseudo_frac"]) == 0.5


# ---------------------------------------------------------------------- #
# agent wiring
# ---------------------------------------------------------------------- #
def test_agent_adds_teacher_builder_only_when_asked(tmp_path):
    _write_cache(tmp_path / "t")
    agent = ParaSSRAgent.__new__(ParaSSRAgent)
    agent._trajectory_sampling = ParaSSRConfig().trajectory_sampling
    agent._config = ParaSSRConfig()
    assert len(agent.get_target_builders()) == 1
    agent._config = replace(ParaSSRConfig(), map_label_source="teacher", kd_teacher_cache=str(tmp_path / "t"))
    builders = agent.get_target_builders()
    assert isinstance(builders[-1], ResMapTeacherTargetBuilder)
    assert builders[-1].get_unique_name().startswith("para_ssr_resmap_teacher_")


def test_config_validation_rejects_incomplete_kd():
    ts = ParaSSRConfig().trajectory_sampling
    with pytest.raises(ValueError, match="kd_teacher_cache"):
        ParaSSRAgent._validate_config(replace(ParaSSRConfig(), kd_mode="feature"), ts)
    with pytest.raises(ValueError, match="kd_readout_ckpt"):
        ParaSSRAgent._validate_config(
            replace(ParaSSRConfig(), kd_mode="readout", kd_teacher_cache="/x"), ts
        )
    with pytest.raises(ValueError, match="kd_mode"):
        ParaSSRAgent._validate_config(replace(ParaSSRConfig(), kd_mode="bogus"), ts)


def test_checkpoint_loader_tolerates_only_the_kd_adapter(tmp_path):
    agent = ParaSSRAgent.__new__(ParaSSRAgent)
    torch.nn.Module.__init__(agent)
    agent.probe = torch.nn.Linear(2, 1)
    agent._checkpoint_path = str(tmp_path / "c.ckpt")
    state = {f"agent.{k}": v for k, v in agent.state_dict().items()}
    state["agent._loss.distiller.adapter.weight"] = torch.ones(1)
    torch.save({"state_dict": state}, tmp_path / "c.ckpt")
    agent.initialize()  # a KD run evaluates without its adapter
    state["agent.other.weight"] = torch.ones(1)
    torch.save({"state_dict": state}, tmp_path / "c.ckpt")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        agent.initialize()


# ---------------------------------------------------------------------- #
# kd_center: distil only the scene-specific part of z (report/19 s3)
# ---------------------------------------------------------------------- #
def test_kd_center_rejects_bad_values_and_the_feature_mode(tmp_path):
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_center="mean"))
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="feature", kd_distance="mse", kd_center="command"))


def test_kd_center_ignores_a_constant_per_command_offset(tmp_path):
    """A student whose z differs from the teacher only by a per-command constant
    costs ~0 with kd_center='command' and >0 without it."""
    torch.manual_seed(0)
    plain = ReadoutDistiller(_kd_config(tmp_path))
    centered = ReadoutDistiller(_kd_config(tmp_path, kd_center="command", kd_center_momentum=0.0))
    reader = plain.__dict__["_reader"]
    bs = 8
    f_t, cmd, _ = _batch(bs, seed=3)
    targets = {"teacher_bev": f_t.half(), "teacher_valid": torch.ones(bs), "command": cmd}
    bev_embed = f_t.float().permute(0, 2, 3, 1).reshape(bs, H * W, C).clone().requires_grad_(True)
    with torch.no_grad():
        z_t = reader.encode(f_t.float(), cmd)
    # the student BEV is the teacher's, so z_S == z_T; add a per-command shift in z
    # by shifting the teacher side instead: same effect on the distance.
    shift = torch.zeros_like(z_t)
    for c in cmd.argmax(1).unique():
        shift[cmd.argmax(1) == c] = 0.5 * torch.randn(1, z_t.shape[-1])
    assert float(plain(bev_embed, targets, 0)["raw"]) == pytest.approx(0.0, abs=1e-5)

    class Shifted(torch.nn.Module):
        def __init__(self, inner, shift):
            super().__init__(); self.inner, self.shift = inner, shift
        def encode(self, bev, cmd, *a, **kw):
            z = self.inner.encode(bev, cmd, *a, **kw)
            return z + self.shift if bev.dtype == torch.float32 and bev.is_leaf is False else z
        def parameters(self):
            return self.inner.parameters()
        @property
        def cfg(self):
            return self.inner.cfg
    # simpler: compare distances on z directly
    d_plain = plain._dist((z_t + shift).transpose(1, 2), z_t.transpose(1, 2))
    valid = torch.ones(bs)
    a, b = centered._centered(z_t + shift, z_t, cmd, valid)
    d_centered = centered._dist(a.transpose(1, 2), b.transpose(1, 2))
    assert float(d_plain.mean()) > 1e-3
    assert float(d_centered.mean()) == pytest.approx(0.0, abs=1e-6)


def test_kd_center_keeps_the_gradient_and_reports_the_plain_distance(tmp_path):
    # the random control: a linear projection, so z varies with the scene (the
    # untrained reader of the fixture pools a random BEV into a near-constant z)
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="random", kd_center="command"))
    for it, seed in enumerate((1, 2, 3)):     # different scenes, so the means are real
        f_t, cmd, _ = _batch(4, seed=seed)
        targets = {"teacher_bev": f_t.half(), "teacher_valid": torch.ones(4), "command": cmd}
        bev_embed = torch.randn(4, H * W, C, requires_grad=True)
        out = d(bev_embed, targets, it)
    assert "raw_uncentered" in out and float(out["raw_uncentered"]) > 0.0
    out["loss"].backward()
    assert bev_embed.grad is not None and bev_embed.grad.abs().sum() > 0
    assert bool(d.center_seen.all()) and d.center_t.abs().sum() > 0


def test_kd_center_means_survive_a_checkpoint(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_center="global"))
    bev_embed, targets = _kd_inputs(bs=4)
    d(bev_embed, targets, 0)
    other = ReadoutDistiller(_kd_config(tmp_path, kd_center="global"))
    other.load_state_dict(d.state_dict())
    assert torch.allclose(other.center_t, d.center_t)
    assert bool(other.center_seen.all()) == bool(d.center_seen.all())


# ---------------------------------------------------------------------- #
# attn_feature: cells weighted by the frozen reader's attention (report/22)
# ---------------------------------------------------------------------- #
def test_attn_feature_needs_a_readout_and_rejects_center(tmp_path):
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="attn_feature", kd_readout_ckpt=None))
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="attn_feature", kd_center="command"))


def test_attn_feature_is_zero_when_student_matches_teacher(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="attn_feature"))
    _, targets = _kd_inputs()
    same = targets["teacher_bev"].float().permute(0, 2, 3, 1).reshape(2, H * W, C)
    assert float(d(same, targets, 0)["raw"]) == pytest.approx(0.0, abs=1e-5)


def test_attn_feature_weights_cells_by_attention(tmp_path):
    """A mismatch only in cells the reader ignores costs less than the same mismatch
    in the cells it attends to."""
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="attn_feature"))
    reader = d.__dict__["_reader"]
    _, targets = _kd_inputs(bs=1)
    f_t = targets["teacher_bev"].float()
    with torch.no_grad():
        _, attn = reader.encode(f_t, targets["command"], return_attn=True)
    w = attn.mean(1)[0]
    order = torch.argsort(w)
    k = max(1, (H * W) // 10)

    def cost(cells):
        f_s = f_t.clone().flatten(2)
        f_s[0, :, cells] = -f_s[0, :, cells]            # flip the sign there: cosine 1 -> -1
        emb = f_s.view(1, C, H, W).permute(0, 2, 3, 1).reshape(1, H * W, C)
        return float(d(emb, targets, 0)["raw"])

    assert cost(order[-k:]) > cost(order[:k])


def test_attn_feature_sends_gradient_to_the_bev_and_not_the_reader(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="attn_feature", kd_adapter=True))
    bev_embed, targets = _kd_inputs()
    out = d(bev_embed, targets, 0)
    out["loss"].backward()
    assert bev_embed.grad is not None and bev_embed.grad.abs().sum() > 0
    assert d.adapter.weight.grad is not None
    assert all(p.grad is None for p in d.__dict__["_reader"].parameters())


# ---------------------------------------------------------------------- #
# sens_feature: only the part of the gap that moves the reader's trajectory
# ---------------------------------------------------------------------- #
def _sens_inputs(bs=2, seed=1):
    f_t, cmd, ego = _batch(bs, seed=seed)
    targets = {"teacher_bev": f_t, "teacher_valid": torch.ones(bs), "command": cmd}
    return f_t, targets, ego


def _as_embed(f):
    return f.permute(0, 2, 3, 1).reshape(f.shape[0], H * W, C)


def _exact_value_jacobian(reader, f_t, cmd, ego):
    """[T*3, H*W, d] d traj / d LN-values of the (single-layer) reader, one sample."""
    leaves = []
    h = reader.layers[0].memory_norm.register_forward_hook(
        lambda m, i, o: leaves.append(o.detach().requires_grad_(True)) or leaves[-1])
    try:
        traj = reader(f_t, cmd, ego)["trajectory"].flatten(1)[0]
        return torch.stack([torch.autograd.grad(traj[j], leaves[0], retain_graph=True)[0][0]
                            for j in range(traj.numel())])
    finally:
        h.remove()


def test_sens_feature_needs_a_readout_rejects_center_and_needs_ego(tmp_path):
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_readout_ckpt=None))
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_center="command"))
    with pytest.raises(ValueError):
        ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_sens_probes=0))
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature"))
    f_t, targets, _ = _sens_inputs()
    with pytest.raises(ValueError):
        d(_as_embed(f_t), targets, 0)                    # the h-readouts read ego late
    assert not any(m._forward_hooks for m in d.__dict__["_reader"].modules())


def test_sens_feature_is_zero_on_a_match_at_any_scale_and_order_one_when_unrelated(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_sens_probes=64))
    f_t, targets, ego = _sens_inputs()
    for scale in (1.0, 2.0, 0.5):
        assert float(d(_as_embed(scale * f_t), targets, 0, ego=ego)["raw"]) == pytest.approx(0.0, abs=1e-6)
    other, _, _ = _sens_inputs(seed=9)
    assert 0.3 < float(d(_as_embed(other), targets, 0, ego=ego)["raw"]) < 5.0     # O(1), not ~400
    assert not any(m._forward_hooks for m in d.__dict__["_reader"].modules())


def test_sens_feature_charges_a_flipped_cell(tmp_path):
    """The failure of a BEV-space Jacobian: behind a LayerNorm, -F_T(u) is in its null
    space.  Measured on the values, flipping cells must cost."""
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_sens_probes=64))
    f_t, targets, ego = _sens_inputs()
    assert float(d(_as_embed(-f_t), targets, 0, ego=ego)["raw"]) > 0.5


def test_sens_feature_weights_cells_by_sensitivity(tmp_path):
    """The same flip costs more in the cells the trajectory is sensitive to."""
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_sens_probes=64))
    reader = d.__dict__["_reader"]
    f_t, targets, ego = _sens_inputs(bs=1)
    J = _exact_value_jacobian(reader, f_t, targets["command"], ego)             # [24, H*W, d]
    order = torch.argsort(J.pow(2).sum((0, 2)))
    k = (H * W) // 10

    def cost(cells):
        f_s = f_t.clone().flatten(2)
        f_s[0, :, cells] = -f_s[0, :, cells]
        return float(d(_as_embed(f_s.view(1, C, H, W)), targets, 0, ego=ego)["raw"])

    assert cost(order[-k:]) > 5 * cost(order[:k])


def test_sens_feature_estimates_the_exact_per_cell_trajectory_change(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_sens_probes=256))
    reader = d.__dict__["_reader"]
    f_t, targets, ego = _sens_inputs(bs=1)
    f_s = f_t + 0.5 * torch.randn(1, C, H, W, generator=torch.Generator().manual_seed(5))
    J = _exact_value_jacobian(reader, f_t, targets["command"], ego)             # [24, H*W, d]
    with torch.no_grad():
        s_scaled = f_s / f_s.norm(dim=1, keepdim=True) * f_t.norm(dim=1, keepdim=True)
        dv = (ReadoutDistiller._values(reader, s_scaled) - ReadoutDistiller._values(reader, f_t))[0, 0]
    exact = (J * dv).sum(-1).pow(2).sum() / (2 * J.pow(2).sum())
    est = float(d(_as_embed(f_s), targets, 0, ego=ego)["raw"])
    assert est == pytest.approx(float(exact), rel=0.15)


def test_sens_feature_sends_gradient_to_the_bev_and_not_the_reader(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature", kd_adapter=True))
    f_t, targets, ego = _sens_inputs()
    bev_embed = (_as_embed(f_t) + 0.1 * torch.randn(2, H * W, C)).requires_grad_(True)
    d(bev_embed, targets, 0, ego=ego)["loss"].backward()
    assert bev_embed.grad is not None and bev_embed.grad.abs().sum() > 0
    assert d.adapter.weight.grad is not None
    assert all(p.grad is None for p in d.__dict__["_reader"].parameters())


def test_sens_feature_is_silent_in_inference_mode_and_keeps_the_global_rng(tmp_path):
    d = ReadoutDistiller(_kd_config(tmp_path, kd_mode="sens_feature"))
    f_t, targets, ego = _sens_inputs()
    with torch.inference_mode():
        assert float(d(-_as_embed(f_t), targets, 0, ego=ego)["raw"]) == 0.0
    torch.manual_seed(7)
    d(-_as_embed(f_t), targets, 0, ego=ego)
    after = torch.rand(3)
    torch.manual_seed(7)
    assert torch.equal(after, torch.rand(3))


def test_loss_passes_the_ego_status_to_sens_feature(tmp_path):
    cfg = _kd_config(
        tmp_path, kd_mode="sens_feature", num_navi_cmd=4,
        grad_balance_target={"plan": 0.5, "distill": 0.5},
        grad_balance_interval=1, grad_balance_momentum=0.0, grad_balance_clamp=(1e-5, 1.0),
        grad_balance_warmup_iters=0, grad_norm_log_interval=0,
        task_loss_weight={"plan": 1.0}, heading_weight=0.5,
        use_det_motion_head=False, use_map_head=False,
    )
    loss = ParaSSRLoss(cfg)
    f_t, targets, ego = _sens_inputs()
    bev_embed = (_as_embed(f_t) + 0.1 * torch.randn(2, H * W, C)).requires_grad_(True)
    targets.update(trajectory_offsets=torch.zeros(2, 8, 3), trajectory_mask=torch.ones(2, 8))
    preds = {"bev_embed": bev_embed, "ego_fut_preds": bev_embed[:, :24, 0].reshape(2, 1, 8, 3).expand(2, 4, 8, 3)}
    features = {"status_feature": torch.cat([targets["command"], ego], dim=1)}
    total, logs = loss(_NoHeadModel().train(), features, targets, preds)
    total.backward()
    assert float(logs["kd/raw"]) > 0 and bev_embed.grad.abs().sum() > 0
