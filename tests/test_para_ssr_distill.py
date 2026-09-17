"""Planning distillation on the navsim PARA-SSR agent.

The cache these exercise is synthesised in ``tmp_path``, so the suite runs
without the 568 GB teacher store.  One test opts into the real cache when it is
present, because the alignment contract is only meaningful against real data.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from navsim.agents.para_ssr.distill import (
    PlanningBEVAdapter,
    ParaSSRTeacherAdapterAgent,
    TeacherAdapterPlanner,
    TeacherCacheMismatch,
    TeacherFeatureStore,
    bev_tokens_to_map,
    build_planning_distillation,
    restrict_dataset_to_stores,
)

REAL_CACHE = "/home/external-user/datasets/teacher_cache"
TOKENS = ("00aabbccddeeff01", "11aabbccddeeff02")


def _write_cache(root, config, tokens=TOKENS, teacher="bevfusion",
                 pc_range=None, shape=None, channels=None):
    """Write a cache with the manifest the real extraction produces."""
    shape = shape or [config.bev_h, config.bev_w]
    channels = config.embed_dims if channels is None else channels
    # mmdet3d LiDAR frame: x forward, y left -- the axis swap the loader undoes.
    sx0, sy0, _, sx1, sy1, _ = config.pc_range
    pc_range = pc_range or [sy0, -sx1, -3.0, sy1, -sx0, 5.0]
    for split in ("train", "val"):
        base = os.path.join(root, teacher, f"cache_{split}_100x100")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "split": split,
                    "target_bev_shape": list(shape),
                    "bev_channels": channels,
                    "point_cloud_range": list(pc_range),
                    "to_student_transform": "student_bev = teacher_bev[:, :, ::-1]",
                },
                fh,
            )
    rng = np.random.default_rng(0)
    for token in tokens:
        shard = os.path.join(root, teacher, "cache_train_100x100", "samples", token[:2])
        os.makedirs(shard, exist_ok=True)
        np.savez(
            os.path.join(shard, token + ".npz"),
            bev_feature=rng.standard_normal((channels, *shape)).astype(np.float16),
        )
    return root


@pytest.fixture()
def config():
    return ParaSSRConfig()


@pytest.fixture()
def cache(tmp_path, config):
    return _write_cache(str(tmp_path / "teacher_cache"), config)


# --------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------- #
def test_bev_tokens_to_map_is_row_major(config):
    tokens = torch.arange(2 * config.bev_h * config.bev_w * 4, dtype=torch.float32)
    tokens = tokens.reshape(2, config.bev_h * config.bev_w, 4)
    maps = bev_tokens_to_map(tokens, (config.bev_h, config.bev_w))
    assert maps.shape == (2, 4, config.bev_h, config.bev_w)
    # cell (r, c) of sample b must be the token at r * bev_w + c
    assert torch.equal(maps[1, :, 3, 7], tokens[1, 3 * config.bev_w + 7])


def test_adapter_starts_as_normalised_identity(config):
    adapter = PlanningBEVAdapter(channels=config.embed_dims)
    x = torch.randn(2, 16, config.embed_dims)
    # fc2 is zero-initialised, so the residual branch contributes nothing yet
    assert torch.allclose(adapter(x), adapter.out_norm(x), atol=1e-6)


# --------------------------------------------------------------------- #
# teacher store
# --------------------------------------------------------------------- #
def test_store_loads_and_flips(cache, config):
    store = TeacherFeatureStore(cache, "bevfusion")
    store.validate_manifest(config)
    assert store.spatial_size() == (config.bev_h, config.bev_w)
    batch = store.load_batch(TOKENS, torch.device("cpu"), torch.float32)
    assert batch.shape == (2, config.embed_dims, config.bev_h, config.bev_w)

    raw = np.load(store.path_for(TOKENS[0]))["bev_feature"]
    assert np.allclose(batch[0].numpy(), raw[:, :, ::-1].astype(np.float32))


def test_store_rejects_mismatched_extent(cache, config):
    """Same shape, different metres: the failure a shape check cannot see."""
    store = TeacherFeatureStore(cache, "bevfusion")
    nuscenes_like = replace(
        config,
        pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
        map_pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
    )
    with pytest.raises(TeacherCacheMismatch):
        store.validate_manifest(nuscenes_like)


def test_store_rejects_mismatched_channels(tmp_path, config):
    root = _write_cache(str(tmp_path / "c"), config, channels=128)
    with pytest.raises(TeacherCacheMismatch):
        TeacherFeatureStore(root, "bevfusion").validate_manifest(config)


def test_store_reports_missing_samples(cache, config):
    store = TeacherFeatureStore(cache, "bevfusion")
    assert store.has_token(TOKENS[0])
    assert not store.has_token("deadbeefdeadbeef")
    with pytest.raises(FileNotFoundError, match="misses"):
        store.load_batch(["deadbeefdeadbeef"], torch.device("cpu"), torch.float32)


class _FakeSceneDataset:
    def __init__(self, tokens):
        class _Loader:
            pass

        self._scene_loader = _Loader()
        self._scene_loader.scene_frames_dicts = {token: {"token": token} for token in tokens}

    def __len__(self):
        return len(self._scene_loader.scene_frames_dicts)


def test_restrict_dataset_drops_uncached_tokens(cache, config):
    store = TeacherFeatureStore(cache, "bevfusion")
    dataset = _FakeSceneDataset(list(TOKENS) + ["deadbeefdeadbeef"])
    dropped = restrict_dataset_to_stores(dataset, [store], name="val")
    assert dropped == 1
    assert set(dataset._scene_loader.scene_frames_dicts) == set(TOKENS)
    assert len(dataset) == len(TOKENS)


def test_store_requires_a_manifest(tmp_path, config):
    os.makedirs(str(tmp_path / "bevfusion"), exist_ok=True)
    with pytest.raises(TeacherCacheMismatch, match="manifest"):
        TeacherFeatureStore(str(tmp_path), "bevfusion").validate_manifest(config)


# --------------------------------------------------------------------- #
# stage 1 -> stage 2
# --------------------------------------------------------------------- #
def _stage1_checkpoint(tmp_path, config, cache):
    cfg = replace(config, use_distill=True, input_target=True,
                  distill_feature_root=cache)
    model = TeacherAdapterPlanner(cfg)
    path = str(tmp_path / "stage1.ckpt")
    torch.save({"state_dict": {"agent.model." + k: v
                               for k, v in model.state_dict().items()}}, path)
    return cfg, model, path


def test_stage1_trains_adapter_and_planner(tmp_path, config, cache):
    cfg, model, _ = _stage1_checkpoint(tmp_path, config, cache)
    bev = TeacherFeatureStore(cache, "bevfusion").load_batch(
        TOKENS, torch.device("cpu"), torch.float32)
    cmd = torch.zeros(2, cfg.num_navi_cmd)
    cmd[:, 1] = 1.0

    before = {n: p.detach().clone() for n, p in model.adapter.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out = model(bev, cmd)
        assert out["ego_fut_preds"].shape == (2, cfg.ego_fut_mode, cfg.fut_ts,
                                              cfg.traj_dims)
        out["ego_fut_preds"].abs().mean().backward()
        opt.step()
    # fc2 is zero-initialised, so fc1 only starts moving after the first step
    assert all(not torch.equal(before[n], p.detach())
               for n, p in model.adapter.named_parameters())


def test_stage2_freezes_adapters_and_reaches_student(tmp_path, config, cache):
    _, _, ckpt = _stage1_checkpoint(tmp_path, config, cache)
    cfg = replace(config, use_distill=True, distill_feature_root=cache,
                  distill_adapter_checkpoints={"bevfusion": ckpt})
    distill = build_planning_distillation(cfg)
    assert distill.loaded_adapter_checkpoints["bevfusion"] == os.path.abspath(ckpt)
    assert not any(p.requires_grad for p in distill.adapters.parameters())

    student = torch.randn(2, cfg.bev_h * cfg.bev_w, cfg.embed_dims,
                          requires_grad=True)
    losses, metrics = distill(student, TOKENS)
    sum(losses.values()).backward()

    assert student.grad.abs().sum() > 0, "distillation must reach the student BEV"
    assert all(p.grad is None for p in distill.adapters.parameters())
    assert "distill_cos/bevfusion" in metrics


def test_stage2_keeps_adapters_in_eval_after_train(tmp_path, config, cache):
    _, _, ckpt = _stage1_checkpoint(tmp_path, config, cache)
    cfg = replace(config, use_distill=True, distill_feature_root=cache,
                  distill_adapter_checkpoints={"bevfusion": ckpt})
    distill = build_planning_distillation(cfg)
    distill.train(True)
    assert all(not a.training for a in distill.adapters.values())


def test_stage2_requires_stage1_checkpoint(config, cache):
    cfg = replace(config, use_distill=True, distill_feature_root=cache,
                  distill_adapter_checkpoints={})
    with pytest.raises(KeyError):
        build_planning_distillation(cfg)


def test_distillation_is_off_by_default(config):
    assert build_planning_distillation(config) is None
    assert not config.needs_scene_token


# --------------------------------------------------------------------- #
# target plumbing
# --------------------------------------------------------------------- #
def test_target_cache_name_changes_with_scene_token(config):
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder

    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    off = ParaSSRTargetBuilder(config, ts).get_unique_name()
    on = ParaSSRTargetBuilder(replace(config, use_distill=True), ts).get_unique_name()
    # a stale target cache without the token would otherwise be reused silently
    assert off != on


# --------------------------------------------------------------------- #
# real cache
# --------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.isdir(os.path.join(REAL_CACHE, "bevfusion")),
    reason="NAVSIM teacher cache not present on this machine",
)
def test_real_cache_matches_the_student_geometry(config):
    store = TeacherFeatureStore(REAL_CACHE, "bevfusion")
    store.validate_manifest(config)
    manifest = store.manifests()["cache_train_100x100"]
    assert manifest["bev_channels"] == config.embed_dims
    assert tuple(manifest["target_bev_shape"]) == (config.bev_h, config.bev_w)


# --------------------------------------------------------------------- #
# launchability
# --------------------------------------------------------------------- #
def test_both_stages_instantiate_through_hydra(tmp_path, config, cache, monkeypatch):
    """navsim builds agents from yaml, so an unregistered agent is unreachable."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "navsim", "planning", "script", "config", "common", "agent",
    )
    monkeypatch.setenv("DISTILL_FEATURE_ROOT", cache)

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        stage1 = instantiate(compose(config_name="para_ssr_teacher_adapter_agent"))
    assert stage1.get_sensor_config().cam_f0 is False, "stage 1 reads no sensors"

    ckpt = str(tmp_path / "hydra_stage1.ckpt")
    torch.save(
        {"state_dict": {"agent." + k: v for k, v in stage1.state_dict().items()}},
        ckpt,
    )
    monkeypatch.setenv("BEVFUSION_ADAPTER_CKPT", ckpt)

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        stage2 = instantiate(compose(config_name="para_ssr_distill_agent"))
    assert stage2._distill is not None
    assert not any(p.requires_grad for p in stage2._distill.adapters.parameters())


def test_stage1_has_no_unused_parameters(config, cache):
    """An unreduced parameter makes DDP abort; single-process runs never show it."""
    cfg = replace(config, use_distill=True, input_target=True,
                  distill_feature_root=cache)
    model = TeacherAdapterPlanner(cfg)
    bev = TeacherFeatureStore(cache, "bevfusion").load_batch(
        TOKENS[:1], torch.device("cpu"), torch.float32)
    cmd = torch.zeros(1, cfg.num_navi_cmd)
    cmd[:, 1] = 1.0
    model(bev, cmd)["ego_fut_preds"].abs().mean().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
    assert dead == [], f"parameters receive no gradient: {dead}"


def test_compute_corridor_mask():
    from navsim.agents.para_ssr.distill import compute_corridor_mask

    trajectories = torch.zeros(2, 8, 3)
    trajectories[:, :, 1] = torch.linspace(0.0, 20.0, 8)
    mask = compute_corridor_mask(
        trajectories=trajectories,
        pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        bev_h=50,
        bev_w=100,
        base_weight=0.1,
    )
    assert mask.shape == (2, 1, 50, 100)
    assert mask.min() >= 0.1
    assert mask.max() <= 1.0
    # Cells near the forward trajectory (x=0, center column ~50) should have high weight
    assert mask[0, 0, 0, 50] > 0.8
    # Distant off-path cells (far right lateral boundary) should have weight close to base_weight
    assert mask[0, 0, 0, 95] < 0.2


def test_stage2_applies_corridor_mask(tmp_path, config, cache):
    _, _, ckpt = _stage1_checkpoint(tmp_path, config, cache)
    cfg = replace(
        config,
        use_distill=True,
        distill_feature_root=cache,
        distill_adapter_checkpoints={"bevfusion": ckpt},
        use_corridor_mask=True,
    )
    distill = build_planning_distillation(cfg)

    student = torch.randn(2, cfg.bev_h * cfg.bev_w, cfg.embed_dims, requires_grad=True)
    trajectories = torch.zeros(2, 8, 3)
    trajectories[:, :, 1] = torch.linspace(0.0, 20.0, 8)

    losses, metrics = distill(student, TOKENS, trajectories=trajectories)
    assert "loss_distill_bevfusion" in losses
    losses["loss_distill_bevfusion"].backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(REAL_CACHE, "resmap")),
    reason="ReSMap teacher cache not present on this machine",
)
def test_real_resmap_cache_matches_geometry_and_loads(config):
    store = TeacherFeatureStore(REAL_CACHE, "resmap")
    store.validate_manifest(config)
    assert store.is_sharded()
    with open(os.path.join(REAL_CACHE, "resmap", "index.json")) as f:
        tok = next(iter(json.load(f).keys()))
    batch = store.load_batch([tok], torch.device("cpu"), torch.float32)
    assert store.spatial_size() == (50, 100)
    assert store.has_token(tok)
    assert not store.has_token("f9a027ce6a5453fa")
    assert batch.shape == (1, config.embed_dims, 50, 100)


def test_stage1_planner_follows_teacher_cache_grid(tmp_path, config):
    """ReSMap-like 50x100 cache must not keep the student's 100x100 positional encoding."""
    root = _write_cache(str(tmp_path / "c"), config, shape=[50, 100])
    cfg = replace(
        config,
        use_distill=True,
        input_target=True,
        distill_feature_root=root,
        teacher_adapter_branch="bevfusion",
    )
    agent = ParaSSRTeacherAdapterAgent(
        cfg, TrajectorySampling(time_horizon=4, interval_length=0.5)
    )
    assert agent.model.planner.bev_h == 50
    assert agent.model.planner.bev_w == 100
    bev = agent._store.load_batch(list(TOKENS), torch.device("cpu"), torch.float32)
    cmd = torch.zeros(len(TOKENS), cfg.num_navi_cmd)
    cmd[:, 1] = 1.0
    out = agent.model(bev, cmd)
    assert out["ego_fut_preds"].shape == (
        len(TOKENS), cfg.ego_fut_mode, cfg.fut_ts, cfg.traj_dims
    )


def test_stage2_resamples_student_when_teacher_grid_differs(tmp_path, config):
    root = _write_cache(str(tmp_path / "c"), config, shape=[50, 100])
    _, _, ckpt = _stage1_checkpoint(tmp_path, config, root)
    cfg = replace(
        config,
        use_distill=True,
        distill_feature_root=root,
        distill_adapter_checkpoints={"bevfusion": ckpt},
        use_corridor_mask=True,
    )
    distill = build_planning_distillation(cfg)
    student = torch.randn(
        2, cfg.bev_h * cfg.bev_w, cfg.embed_dims, requires_grad=True
    )
    trajectories = torch.zeros(2, 8, 3)
    trajectories[:, :, 1] = torch.linspace(0.0, 20.0, 8)
    losses, _metrics = distill(student, TOKENS, trajectories=trajectories)
    losses["loss_distill_bevfusion"].backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
