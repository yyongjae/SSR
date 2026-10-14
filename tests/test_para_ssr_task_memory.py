"""Planner task-memory layouts (report/41): motion mode tokens and map segments.

Exercises the real ``ParaSSRPlannerHead`` at small width on CPU with an identity
transformer.  The reference formulas are written out independently from the
head's own sub-modules so that the default layout is checked against the
original pooled memories, and every new layout against its specification.
"""
import os
from dataclasses import fields

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.modules.planner_head import ParaSSRPlannerHead, validate_plan_memory_options
from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel

C = 32
BEV_H, BEV_W = 4, 5
PC_RANGE = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
COMBOS = [("mean", 1), ("mode_tokens", 1), ("mean", 4), ("mode_tokens", 4)]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_YAML_DIR = os.path.join(REPO, "navsim", "planning", "script", "config", "common", "agent")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _head(mode="mean", segments=1, *, anchor_file=None, use_task_interaction=True, seed=0, **kw):
    torch.manual_seed(seed)
    options = {} if mode is None else dict(plan_det_memory_mode=mode, plan_map_memory_segments=segments)
    return ParaSSRPlannerHead(
        transformer=torch.nn.Identity(), bev_h=BEV_H, bev_w=BEV_W, embed_dims=C, num_heads=4,
        feedforward_channels=64, pc_range=PC_RANGE, use_task_interaction=use_task_interaction,
        plan_anchor_file=anchor_file, **options, **kw,
    )


def _inputs(B=2, Q=3, M=2, V=3, P=4, *, c_det=C, c_motion=C, c_map=C, layers=2, seed=1, leaf=False):
    """Synthetic head outputs.  ``leaf=True`` makes every tensor an independent leaf."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).requires_grad_(leaf)
    det_out = {
        "det_hidden": r(B, Q, c_det),
        "motion_hidden": r(B, Q, M, c_motion),
        "all_bbox_preds": r(layers, B, Q, 10, scale=10.0),
        "all_cls_scores": r(layers, B, Q, 7),
    }
    map_out = {
        "map_point_hidden": r(B, V, P, c_map),
        "all_map_pts_preds": torch.rand(layers, B, V, P, 2, generator=g).requires_grad_(leaf),
        "all_map_cls_scores": r(layers, B, V, 4),
    }
    return det_out, map_out


def _bev(B=2, seed=2, leaf=False):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, BEV_H * BEV_W, C, generator=g).requires_grad_(leaf)


def _plan(head, bev, det_out, map_out):
    B = bev.shape[0]
    cmd = torch.eye(4)[torch.arange(B) % 4]
    ego = torch.linspace(-1.0, 1.0, B * 4).reshape(B, 4)
    return head.plan_from_bev(bev, cmd, det_out, map_out, ego_status=ego)


def _reference_metadata(head, det_out, map_out):
    """Original (pre-change) metadata formulas, written independently."""
    det_xy = det_out["all_bbox_preds"][-1, ..., :2].detach()
    det_xy = (det_xy - det_xy.new_tensor(PC_RANGE[:2])) / det_xy.new_tensor((64.0, 32.0))
    det_score = det_out["all_cls_scores"][-1].detach().sigmoid().amax(-1, keepdim=True)
    map_xy = map_out["all_map_pts_preds"][-1].detach()
    map_score = map_out["all_map_cls_scores"][-1].detach().sigmoid().amax(-1, keepdim=True)
    return det_xy, det_score, map_xy, map_score


def _reference_original(head, det_out, map_out):
    """The exact pre-change memory construction (mean over modes, mean over points)."""
    det_xy, det_score, map_xy, map_score = _reference_metadata(head, det_out, map_out)
    det_memory = head.det_memory_norm(
        head.det_projection(det_out["det_hidden"])
        + head.motion_projection(det_out["motion_hidden"].mean(dim=2))
    )
    map_memory = head.map_memory_norm(head.map_projection(map_out["map_point_hidden"].mean(dim=2)))
    return {
        "det_memory": det_memory,
        "det_position": head.det_position_encoder(det_xy),
        "det_confidence": head.det_confidence_encoder(det_score),
        "map_memory": map_memory,
        "map_position": head.map_position_encoder(map_xy).mean(dim=2),
        "map_confidence": head.map_confidence_encoder(map_score),
    }


def _assert_memories_close(actual, expected, **tol):
    assert set(actual) == set(expected) == {
        "det_memory", "det_position", "det_confidence", "map_memory", "map_position", "map_confidence"}
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        torch.testing.assert_close(actual[key], expected[key], msg=key, **tol)


@pytest.fixture
def anchor_file(tmp_path):
    g = np.random.default_rng(0)
    anchors = np.cumsum(g.normal(size=(8, 8, 3)).astype(np.float32) * [2.0, 0.3, 0.05], axis=1)
    path = tmp_path / "anchors.npy"
    np.save(path, anchors.astype(np.float32))
    return str(path)


# --------------------------------------------------------------------------- #
# default compatibility
# --------------------------------------------------------------------------- #
def test_default_options_reproduce_the_original_memories():
    det_out, map_out = _inputs()
    implicit = _head(mode=None).eval()
    explicit = _head("mean", 1).eval()
    assert implicit.plan_det_memory_mode == "mean" and implicit.plan_map_memory_segments == 1
    with torch.no_grad():
        for head in (implicit, explicit):
            _assert_memories_close(head.prepare_task_memories(det_out, map_out),
                                   _reference_original(head, det_out, map_out), rtol=0, atol=0)
        # same parameters (same seed) -> same planner output with and without the kwargs
        bev = _bev()
        out_implicit = _plan(implicit, bev, det_out, map_out)
        out_explicit = _plan(explicit, bev, det_out, map_out)
    torch.testing.assert_close(out_implicit["ego_fut_preds"], out_explicit["ego_fut_preds"], rtol=0, atol=0)
    assert torch.isfinite(out_implicit["ego_fut_preds"]).all()


# --------------------------------------------------------------------------- #
# shapes for the four combinations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode, segments", COMBOS)
@pytest.mark.parametrize("B, Q, M, V, P", [(2, 300, 6, 100, 20), (1, 2, 3, 2, 8), (3, 5, 1, 4, 4)])
def test_memory_shapes_for_each_combination(mode, segments, B, Q, M, V, P):
    head = _head(mode, segments, det_embed_dims=24, motion_embed_dims=16, map_embed_dims=40).eval()
    det_out, map_out = _inputs(B, Q, M, V, P, c_det=24, c_motion=16, c_map=40)
    with torch.no_grad():
        mem = head.prepare_task_memories(det_out, map_out)
    n_det = Q * M if mode == "mode_tokens" else Q
    n_map = V * segments
    for key in ("det_memory", "det_position", "det_confidence"):
        assert mem[key].shape == (B, n_det, C), key
    for key in ("map_memory", "map_position", "map_confidence"):
        assert mem[key].shape == (B, n_map, C), key


# --------------------------------------------------------------------------- #
# motion mode tokens: order and information
# --------------------------------------------------------------------------- #
def test_mode_tokens_keep_every_mode_in_object_major_order():
    Q, M = 2, 3
    head = _head("mode_tokens", 1).eval()
    det_out, map_out = _inputs(B=2, Q=Q, M=M)
    with torch.no_grad():
        mem = head.prepare_task_memories(det_out, map_out)
        ref = _reference_original(head, det_out, map_out)
        det_content = head.det_projection(det_out["det_hidden"])
        motion_content = head.motion_projection(det_out["motion_hidden"])
        for q in range(Q):
            for m in range(M):
                i = q * M + m
                expected = head.det_memory_norm(det_content[:, q] + motion_content[:, q, m])
                torch.testing.assert_close(mem["det_memory"][:, i], expected)
                torch.testing.assert_close(mem["det_position"][:, i], ref["det_position"][:, q], rtol=0, atol=0)
                torch.testing.assert_close(mem["det_confidence"][:, i], ref["det_confidence"][:, q], rtol=0, atol=0)
        # modes of one object differ in content, unless the modes are identical
        assert not torch.allclose(mem["det_memory"][:, 0], mem["det_memory"][:, 1])
        det_out["motion_hidden"] = det_out["motion_hidden"][:, :, :1].expand(-1, -1, M, -1)
        same = head.prepare_task_memories(det_out, map_out)["det_memory"].reshape(2, Q, M, C)
        torch.testing.assert_close(same, same[:, :, :1].expand(-1, -1, M, -1))
        # map side is untouched by the detection option
        for key in ("map_memory", "map_position", "map_confidence"):
            torch.testing.assert_close(mem[key], ref[key], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# map segments: pooling, position order, locality
# --------------------------------------------------------------------------- #
def test_map_segments_pool_contiguous_points_in_element_major_order():
    V, P, S = 3, 8, 4
    L = P // S
    head = _head("mean", S).eval()
    det_out, map_out = _inputs(B=2, V=V, P=P)
    hidden, xy = map_out["map_point_hidden"], map_out["all_map_pts_preds"][-1]
    with torch.no_grad():
        mem = head.prepare_task_memories(det_out, map_out)
        ref = _reference_original(head, det_out, map_out)
        for v in range(V):
            for s in range(S):
                j = v * S + s
                sl = slice(s * L, (s + 1) * L)
                expected_mem = head.map_memory_norm(head.map_projection(hidden[:, v, sl].mean(dim=1)))
                torch.testing.assert_close(mem["map_memory"][:, j], expected_mem)
                expected_pos = head.map_position_encoder(xy[:, v, sl]).mean(dim=1)
                torch.testing.assert_close(mem["map_position"][:, j], expected_pos)
                # encode-then-average, not average-then-encode (the encoder is non-linear)
                assert not torch.allclose(mem["map_position"][:, j],
                                          head.map_position_encoder(xy[:, v, sl].mean(dim=1)), atol=1e-4)
                torch.testing.assert_close(mem["map_confidence"][:, j], ref["map_confidence"][:, v], rtol=0, atol=0)
        # det side is untouched by the map option
        for key in ("det_memory", "det_position", "det_confidence"):
            torch.testing.assert_close(mem[key], ref[key], rtol=0, atol=0)
        # perturbing one segment's hidden points changes only that memory token
        v, s = 1, 2
        perturbed = hidden.clone()
        perturbed[:, v, s * L:(s + 1) * L] += 3.0
        mem2 = head.prepare_task_memories(det_out, {**map_out, "map_point_hidden": perturbed})
        changed = ~torch.isclose(mem2["map_memory"], mem["map_memory"], atol=1e-6).all(-1)
        expected_changed = torch.zeros(2, V * S, dtype=torch.bool)
        expected_changed[:, v * S + s] = True
        assert torch.equal(changed, expected_changed)
        torch.testing.assert_close(mem2["map_position"], mem["map_position"], rtol=0, atol=0)
        # and perturbing one segment's XY changes only that position token
        xy2 = map_out["all_map_pts_preds"].clone()
        xy2[-1, :, v, s * L:(s + 1) * L] += 0.5
        mem3 = head.prepare_task_memories(det_out, {**map_out, "all_map_pts_preds": xy2})
        changed = ~torch.isclose(mem3["map_position"], mem["map_position"], atol=1e-6).all(-1)
        assert torch.equal(changed, expected_changed)


def test_point_tokens_are_the_segment_count_equal_to_points():
    V, P = 2, 4
    head = _head("mean", P).eval()
    det_out, map_out = _inputs(B=1, V=V, P=P)
    with torch.no_grad():
        mem = head.prepare_task_memories(det_out, map_out)
        expected = head.map_memory_norm(head.map_projection(map_out["map_point_hidden"])).flatten(1, 2)
    torch.testing.assert_close(mem["map_memory"], expected)
    assert mem["map_memory"].shape == (1, V * P, C)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode, segments", COMBOS)
@pytest.mark.parametrize("v2", [False, True])
def test_hidden_content_keeps_gradients_and_metadata_is_detached(mode, segments, v2, anchor_file):
    head = _head(mode, segments, anchor_file=anchor_file if v2 else None).train()
    det_out, map_out = _inputs(leaf=True)
    bev = _bev(leaf=True)
    out = _plan(head, bev, det_out, map_out)
    weights = torch.linspace(0.3, 1.7, out["ego_fut_preds"].numel()).reshape(out["ego_fut_preds"].shape)
    loss = (out["ego_fut_preds"] * weights).sum()
    if v2:
        loss = loss + (out["im_rewards"] * torch.linspace(-1, 1, out["im_rewards"].numel()).reshape(out["im_rewards"].shape)).sum()
    loss.backward()
    for name, tensor in (("det_hidden", det_out["det_hidden"]), ("motion_hidden", det_out["motion_hidden"]),
                         ("map_point_hidden", map_out["map_point_hidden"]), ("bev", bev)):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name
    for name in ("all_bbox_preds", "all_cls_scores"):
        assert det_out[name].is_leaf and det_out[name].grad is None, name
    for name in ("all_map_pts_preds", "all_map_cls_scores"):
        assert map_out[name].is_leaf and map_out[name].grad is None, name
    # metadata encoder parameters still learn
    for enc in (head.det_position_encoder, head.det_confidence_encoder,
                head.map_position_encoder, head.map_confidence_encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_every_mode_and_segment_receives_gradient():
    head = _head("mode_tokens", 4).train()
    det_out, map_out = _inputs(B=1, Q=2, M=3, V=2, P=8, leaf=True)
    out = _plan(head, _bev(B=1), det_out, map_out)
    (out["ego_fut_preds"] * torch.linspace(0.1, 2.0, out["ego_fut_preds"].numel()).reshape(out["ego_fut_preds"].shape)).sum().backward()
    assert (det_out["motion_hidden"].grad.abs().sum(-1) > 0).all()        # every (q, m)
    assert (map_out["map_point_hidden"].grad.abs().sum(-1) > 0).all()     # every (v, p)


# --------------------------------------------------------------------------- #
# non-contiguous inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode, segments", COMBOS)
def test_non_contiguous_inputs_match_contiguous(mode, segments):
    head = _head(mode, segments).eval()
    det_out, map_out = _inputs(B=2, Q=3, M=2, V=3, P=4)

    def view(t):
        # a strided slice of a wider tensor: equal in value, non-contiguous in memory
        wide = torch.stack([t, t * 0.0 - 5.0], dim=-1)     # [..., 2]
        return wide[..., 0]

    nc_det = {k: view(v) for k, v in det_out.items()}
    nc_map = {k: view(v) for k, v in map_out.items()}
    for d in (nc_det, nc_map):
        for k, v in d.items():
            assert not v.is_contiguous(), k
            torch.testing.assert_close(v, det_out.get(k, map_out.get(k)), rtol=0, atol=0)
    with torch.no_grad():
        _assert_memories_close(head.prepare_task_memories(nc_det, nc_map),
                               head.prepare_task_memories(det_out, map_out), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# invalid options and inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw, message", [
    ({"plan_det_memory_mode": "weighted"}, "plan_det_memory_mode"),
    ({"plan_det_memory_mode": None}, "plan_det_memory_mode"),
    ({"plan_map_memory_segments": True}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": False}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": 0}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": -4}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": 4.0}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": "4"}, "plan_map_memory_segments"),
])
@pytest.mark.parametrize("use_task_interaction", [True, False])
def test_constructor_rejects_invalid_options(kw, message, use_task_interaction):
    with pytest.raises(ValueError, match=message):
        _head(mode=None, use_task_interaction=use_task_interaction, **kw)


@pytest.mark.parametrize("segments, P", [(3, 20), (8, 4), (16, 20), (5, 4)])
def test_segments_must_divide_the_real_point_count(segments, P):
    head = _head("mean", segments).eval()
    det_out, map_out = _inputs(V=2, P=P)
    with pytest.raises(ValueError, match="plan_map_memory_segments"):
        head.prepare_task_memories(det_out, map_out)
    with pytest.raises(ValueError, match="plan_map_memory_segments"):
        validate_plan_memory_options("mean", segments, P)


@pytest.mark.parametrize("mode, segments", COMBOS)
@pytest.mark.parametrize("corrupt", [
    lambda d, m: d.__setitem__("all_bbox_preds", d["all_bbox_preds"][:, :, :2]),            # Q mismatch
    lambda d, m: d.__setitem__("all_cls_scores", d["all_cls_scores"][:, :1]),               # B mismatch
    lambda d, m: m.__setitem__("all_map_pts_preds", m["all_map_pts_preds"][:, :, :, :3]),   # P mismatch
    lambda d, m: m.__setitem__("all_map_cls_scores", m["all_map_cls_scores"][:, :, :2]),    # V mismatch
    lambda d, m: m.__setitem__("map_point_hidden", m["map_point_hidden"][:1]),             # batch mismatch
    lambda d, m: d.__setitem__("motion_hidden", d["motion_hidden"][:, :2]),                 # object mismatch
    lambda d, m: d.__setitem__("motion_hidden", d["motion_hidden"][:, :, 0]),               # rank
])
def test_mismatched_metadata_fails_fast(mode, segments, corrupt):
    head = _head(mode, segments).eval()
    det_out, map_out = _inputs(B=2, Q=3, M=2, V=3, P=4)
    corrupt(det_out, map_out)
    with pytest.raises(ValueError):
        head.prepare_task_memories(det_out, map_out)


# --------------------------------------------------------------------------- #
# interaction off
# --------------------------------------------------------------------------- #
def test_interaction_off_keeps_bev_only_path_and_builds_no_task_modules():
    head = _head("mode_tokens", 4, use_task_interaction=False).eval()
    assert head.plan_det_memory_mode == "mode_tokens" and head.plan_map_memory_segments == 4
    for name in ("det_projection", "motion_projection", "map_projection", "det_memory_norm",
                 "map_memory_norm", "det_position_encoder", "map_position_encoder",
                 "det_confidence_encoder", "map_confidence_encoder"):
        assert not hasattr(head, name), name
    assert not any("det" in k or "map" in k for k in head.state_dict())
    with torch.no_grad():
        out = _plan(head, _bev(), None, None)
    assert out["ego_fut_preds"].shape == (2, 4, 8, 3) and torch.isfinite(out["ego_fut_preds"]).all()
    torch.testing.assert_close(out["ego_fut_preds"], _plan(_head(use_task_interaction=False).eval(), _bev(), None, None)["ego_fut_preds"])
    with pytest.raises(ValueError, match="use_task_interaction"):
        head.prepare_task_memories(*_inputs())


# --------------------------------------------------------------------------- #
# checkpoint compatibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v2", [False, True])
def test_state_dict_is_identical_across_memory_options(v2, anchor_file):
    heads = {combo: _head(*combo, anchor_file=anchor_file if v2 else None) for combo in COMBOS}
    base = heads[("mean", 1)]
    base_shapes = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    assert base_shapes, "state dict must not be empty"
    for combo, head in heads.items():
        shapes = {k: tuple(v.shape) for k, v in head.state_dict().items()}
        assert shapes == base_shapes, combo
        head.load_state_dict(base.state_dict(), strict=True)
    # the options are plain attributes, never buffers or parameters
    assert not any("memory_mode" in n or "memory_segments" in n
                   for n, _ in list(base.named_buffers()) + list(base.named_parameters()))


# --------------------------------------------------------------------------- #
# v1 / v2 smoke on the combined configuration
# --------------------------------------------------------------------------- #
def test_v1_single_query_combined_forward_backward():
    head = _head("mode_tokens", 4).train()
    det_out, map_out = _inputs(B=2, Q=4, M=3, V=3, P=8, leaf=True)
    out = _plan(head, _bev(leaf=True), det_out, map_out)
    assert set(out) == {"bev_embed", "scene_query", "token_attn", "ego_fut_preds"}
    assert out["ego_fut_preds"].shape == (2, 4, 8, 3) and out["scene_query"].shape == (1, 2, C)
    out["ego_fut_preds"].square().sum().backward()
    assert torch.isfinite(out["ego_fut_preds"]).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_v2_anchor_queries_combined_forward_backward(anchor_file):
    head = _head("mode_tokens", 4, anchor_file=anchor_file).train()
    det_out, map_out = _inputs(B=2, Q=4, M=3, V=3, P=8, leaf=True)
    out = _plan(head, _bev(leaf=True), det_out, map_out)
    K = head.anchor_planner.num_anchors
    assert out["trajectory"].shape == (2, 8, 3) and out["trajectory_offset"].shape == (2, K, 8, 3)
    assert out["im_rewards"].shape == (2, K) and out["ego_fut_preds"].shape == (2, 4, 8, 3)
    assert out["scene_query"].shape == (K, 2, C)
    (out["trajectory_offset"].square().sum() + out["im_rewards"].log().sum() + out["sim_rewards"].sum()).backward()
    assert torch.isfinite(out["trajectory"]).all()
    assert det_out["motion_hidden"].grad is not None and torch.isfinite(det_out["motion_hidden"].grad).all()
    assert map_out["map_point_hidden"].grad is not None and torch.isfinite(map_out["map_point_hidden"].grad).all()
    # evaluation-only rescoring also runs over the enlarged memories
    head.plan_rescore_refined = True
    head.eval()
    with torch.no_grad():
        out = _plan(head, _bev(), *_inputs(B=2, Q=4, M=3, V=3, P=8))
    assert out["plan_rescore_final_rewards"].shape == (2, K)


# --------------------------------------------------------------------------- #
# config wiring: dataclass, YAML, Hydra, agent validation, model
# --------------------------------------------------------------------------- #
def test_dataclass_and_yaml_defaults_agree():
    cfg = ParaSSRConfig()
    assert cfg.plan_det_memory_mode == "mean" and cfg.plan_map_memory_segments == 1
    assert {f.name for f in fields(ParaSSRConfig)} >= {"plan_det_memory_mode", "plan_map_memory_segments"}
    with open(os.path.join(AGENT_YAML_DIR, "para_ssr_agent.yaml")) as f:
        doc = yaml.safe_load(f)["config"]
    assert doc["plan_det_memory_mode"] == cfg.plan_det_memory_mode
    assert doc["plan_map_memory_segments"] == cfg.plan_map_memory_segments


@pytest.mark.parametrize("mode, segments", COMBOS)
def test_hydra_overrides_compose_without_plus_prefix(mode, segments):
    with initialize_config_dir(config_dir=AGENT_YAML_DIR, version_base=None):
        cfg = compose(config_name="para_ssr_agent", overrides=[
            f"config.plan_det_memory_mode={mode}", f"config.plan_map_memory_segments={segments}"])
    config = instantiate(cfg.config)
    assert isinstance(config, ParaSSRConfig)
    assert config.plan_det_memory_mode == mode and config.plan_map_memory_segments == segments
    assert isinstance(config.plan_map_memory_segments, int)
    ParaSSRAgent._validate_config(config, config.trajectory_sampling)


@pytest.mark.parametrize("override, message", [
    ({"plan_det_memory_mode": "softmax"}, "plan_det_memory_mode"),
    ({"plan_map_memory_segments": True}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": 0}, "plan_map_memory_segments"),
    ({"plan_map_memory_segments": 3}, "plan_map_memory_segments"),      # 20 % 3 != 0
    ({"plan_map_memory_segments": 40}, "plan_map_memory_segments"),     # S > P
])
def test_agent_validation_rejects_bad_memory_options(override, message):
    cfg = ParaSSRConfig(**override)
    with pytest.raises(ValueError, match=message):
        ParaSSRAgent._validate_config(cfg, cfg.trajectory_sampling)


def test_agent_validation_accepts_point_tokens_and_skips_divisibility_without_interaction():
    cfg = ParaSSRConfig(plan_map_memory_segments=20)
    ParaSSRAgent._validate_config(cfg, cfg.trajectory_sampling)
    # Without interaction the format is still checked but P-divisibility is not required.
    off = ParaSSRConfig(use_task_interaction=False, plan_map_memory_segments=3)
    ParaSSRAgent._validate_config(off, off.trajectory_sampling)
    bad = ParaSSRConfig(use_task_interaction=False, plan_det_memory_mode="nope")
    with pytest.raises(ValueError, match="plan_det_memory_mode"):
        ParaSSRAgent._validate_config(bad, bad.trajectory_sampling)


def _small_model_config(**overrides):
    return ParaSSRConfig(
        use_lidar=False, image_architecture="resnet18", backbone_pretrained=False,
        image_scale=1.0, crop_top=0, bev_h=8, bev_w=8, embed_dims=16, num_heads=2,
        ffn_channels=32, encoder_num_layers=1, encoder_num_points_in_pillar=2,
        encoder_num_points_sca=2, encoder_attn_dropout=0.0, encoder_ffn_dropout=0.0,
        latent_num_layers=1, num_scenes=2, num_query=4, max_agents=4, det_num_decoder_layers=1,
        map_num_vec=3, map_num_pts_per_vec=4, map_num_decoder_layers=1, use_grid_mask=False,
        **overrides,
    )


def test_model_passes_both_options_to_the_planner_head(monkeypatch):
    import navsim.agents.para_ssr.para_ssr_model as model_module

    captured = {}
    real = model_module.ParaSSRPlannerHead

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(model_module, "ParaSSRPlannerHead", spy)
    model = ParaSSRModel(_small_model_config(plan_det_memory_mode="mode_tokens", plan_map_memory_segments=2))
    assert captured["plan_det_memory_mode"] == "mode_tokens" and captured["plan_map_memory_segments"] == 2
    assert model.pts_bbox_head.plan_det_memory_mode == "mode_tokens"
    assert model.pts_bbox_head.plan_map_memory_segments == 2
    captured.clear()
    ParaSSRModel(_small_model_config())
    assert captured["plan_det_memory_mode"] == "mean" and captured["plan_map_memory_segments"] == 1


def test_model_state_dict_is_identical_across_memory_options():
    base = ParaSSRModel(_small_model_config())
    keys = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    for mode, segments in COMBOS:
        other = ParaSSRModel(_small_model_config(plan_det_memory_mode=mode, plan_map_memory_segments=min(segments, 4)))
        assert {k: tuple(v.shape) for k, v in other.state_dict().items()} == keys, (mode, segments)
        other.load_state_dict(base.state_dict(), strict=True)
