"""W&B logger wiring: run config + auto tags derived from the Hydra training config."""
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from navsim.planning.training.wandb_logging import (
    auto_tags_from_cfg,
    build_loggers,
    hparams_from_cfg,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CFG_DIR = os.path.join(REPO, "navsim", "planning", "script", "config", "training")
COMBOS = [("mean", 1), ("mode_tokens", 1), ("mean", 4), ("mode_tokens", 4)]


def _cfg(tmp_path, *overrides):
    with initialize_config_dir(config_dir=TRAIN_CFG_DIR, version_base="1.2"):
        return compose(config_name="default_training", overrides=[
            "agent=para_ssr_agent", "experiment_name=unit_wb",
            f"output_dir={tmp_path}", *overrides])


@pytest.mark.parametrize("mode, segments", COMBOS)
def test_auto_tags_and_hparams_follow_overrides(tmp_path, mode, segments):
    cfg = _cfg(tmp_path, f"agent.config.plan_det_memory_mode={mode}",
               f"agent.config.plan_map_memory_segments={segments}")
    tags = auto_tags_from_cfg(cfg)
    assert f"det_mem:{mode}" in tags and f"map_seg:{segments}" in tags
    assert any(t.startswith("backbone:") for t in tags)
    assert "v2-anchor" not in tags
    hp = hparams_from_cfg(cfg)
    assert hp["experiment_name"] == "unit_wb"
    assert hp["agent"]["config"]["plan_det_memory_mode"] == mode
    assert hp["agent"]["config"]["plan_map_memory_segments"] == segments
    assert isinstance(hp["agent"], dict) and isinstance(hp["trainer"], dict)  # plain, not OmegaConf


def test_auto_tags_mark_v2_anchor(tmp_path):
    cfg = _cfg(tmp_path, "agent.config.plan_anchor=true")
    assert "v2-anchor" in auto_tags_from_cfg(cfg)


def test_auto_tags_empty_without_agent_config():
    assert auto_tags_from_cfg(OmegaConf.create({"agent": {}})) == []
    assert auto_tags_from_cfg(OmegaConf.create({})) == []


def test_disabled_wandb_keeps_tensorboard_only(tmp_path):
    cfg = _cfg(tmp_path, "wandb.enable=false")
    assert [type(l).__name__ for l in build_loggers(cfg)] == ["TensorBoardLogger"]


def test_wandb_run_receives_config_and_merged_tags(tmp_path):
    # mode=offline needs no network or login and, unlike "disabled", keeps run.tags.
    # The offline run is written under tmp_path and discarded with it.
    cfg = _cfg(tmp_path, "wandb.mode=offline", "wandb.tags=[para-ssr,navsim]",
               "agent.config.plan_det_memory_mode=mode_tokens", "agent.config.plan_map_memory_segments=4")
    loggers = build_loggers(cfg)
    assert [type(l).__name__ for l in loggers] == ["TensorBoardLogger", "SafeWandbLogger"]
    run = loggers[-1].experiment
    try:
        cfg_dict = dict(run.config)
        assert cfg_dict["agent"]["config"]["plan_det_memory_mode"] == "mode_tokens"
        assert cfg_dict["agent"]["config"]["plan_map_memory_segments"] == 4
        assert cfg_dict["experiment_name"] == "unit_wb"
        assert set(run.tags) >= {"para-ssr", "navsim", "det_mem:mode_tokens", "map_seg:4"}
        assert len(run.tags) == len(set(run.tags))
    finally:
        run.finish()
