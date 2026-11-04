from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple
import fnmatch
import hashlib
import json
import os
import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging, torch
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml

import navsim

REPO_ROOT = Path(__file__).resolve().parents[3]


def check_navsim_import(navsim_file: Optional[str], repo_root: Path = REPO_ROOT) -> None:
    """The package must come from this script's repo.  The hydra config is read relative to this script, so with
    another checkout on the path (an editable install, no PYTHONPATH) that checkout's model would train with this
    checkout's config, without any error."""
    if navsim_file is None or Path(repo_root).resolve() not in Path(navsim_file).resolve().parents:
        raise RuntimeError(f"navsim is imported from {navsim_file}, not from this repo {repo_root}: set "
                           f"PYTHONPATH={repo_root}")


check_navsim_import(navsim.__file__)     # before any other navsim module is imported

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.wandb_logging import build_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


def _worker_init(worker_id: int) -> None:
    """Keep every dataloader worker single-threaded.

    A worker is already a process, so its libraries' own thread pools add no
    parallelism -- they only oversubscribe the machine.  OpenCV defaults to one
    thread per core and is inherited through the fork, so N workers x C cores
    threads fight over C cores: measured here at 2,685 threads on 32 cores and
    491k context switches/second, which starved the GPUs to ~60% and made a
    step 18x slower than its compute cost.  PyTorch already pins its own
    intra-op threads in workers; OpenCV and the OpenMP runtimes do not.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
NAVTEST_FILTER = REPO_ROOT / "navsim/planning/script/config/common/scene_filter/navtest.yaml"
# test sets of the challenge: never training data
TEST_SCENE_FILTERS = ("navtest", "navhard*", "private*", "warmup_test*")


def check_launch_route(env: Mapping[str, str]) -> None:
    """A plain call of this script gets the config defaults, which are not the recipe of any experiment (another
    backbone, BEV grid, map head, no KD, ...).  Runs come through scripts/experiment.sh (SSR_EXPERIMENT_SHA) or a script
    that states it knows (SSR_ALLOW_DIRECT=1)."""
    if not env.get("SSR_EXPERIMENT_SHA") and env.get("SSR_ALLOW_DIRECT") != "1":
        raise RuntimeError("run_training.py was started without a recipe: use scripts/experiment.sh train, or set "
                           "SSR_ALLOW_DIRECT=1 if the overrides are complete")


def check_log_splits(train_logs: Iterable[str], val_logs: Iterable[str], test_logs: Iterable[str]) -> None:
    train, val, test = set(train_logs), set(val_logs), set(test_logs)
    for name, shared in (("train / val", train & val), ("train / test", train & test), ("val / test", val & test)):
        if shared:
            raise RuntimeError(f"log split: {len(shared)} logs are in both {name}, e.g. {sorted(shared)[:3]}")


def check_scene_filter_name(name: Optional[str]) -> None:
    if name is not None and any(fnmatch.fnmatch(str(name), p) for p in TEST_SCENE_FILTERS):
        raise RuntimeError(f"scene_filter={name} is a test set: refused for training")


def navtest_tokens(path: Path = NAVTEST_FILTER) -> Set[str]:
    with open(path) as f:
        return {str(t) for t in yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["tokens"]}


def check_and_record_train_tokens(tokens: Sequence[str], output_dir: Optional[str], is_rank_zero: bool = True,
                                  test_tokens: Optional[Set[str]] = None) -> Dict[str, object]:
    """No final training token may be a navtest token (this holds with the refiner off too, unlike the refiner's packed
    row check).  Rank 0 writes the token list's sha256 and count to <output_dir>/train_tokens.json, which makes two runs'
    training sets comparable."""
    tokens = [str(t) for t in tokens]
    shared = set(tokens) & (navtest_tokens() if test_tokens is None else test_tokens)
    if shared:
        raise RuntimeError(f"{len(shared)} training tokens are navtest tokens, e.g. {sorted(shared)[:3]}")
    rec = {"train_token_sha256": hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest(), "count": len(tokens)}
    if is_rank_zero and output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "train_tokens.json.tmp"
        tmp.write_text(json.dumps(rec) + "\n")
        os.replace(tmp, out / "train_tokens.json")
    return rec


def dataset_tokens(data) -> Sequence[str]:
    """tokens of a Dataset (through its scene loader) or a CacheOnlyDataset."""
    return data._scene_loader.tokens if hasattr(data, "_scene_loader") else data.tokens


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
        train_scene_filter.log_names = sorted(
            set(train_scene_filter.log_names) & set(cfg.train_logs)
        )
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
        val_scene_filter.log_names = sorted(
            set(val_scene_filter.log_names) & set(cfg.val_logs)
        )
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    return train_data, val_data


@hydra.main(version_base="1.2", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    check_launch_route(os.environ)
    check_log_splits(cfg.train_logs, cfg.val_logs, cfg.test_logs)
    check_scene_filter_name(HydraConfig.get().runtime.choices.get("scene_filter"))
    seed = int(cfg.get("seed", 0))
    logger.info("Global Seed set to %d", seed)
    pl.seed_everything(seed, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    if hasattr(agent, "filter_datasets_to_teacher_cache"):
        # aux_distill BEV distillation: keep only training tokens every teacher
        # cache has (ReSMap covers navtrain train_logs only).  No-op when it is off.
        agent.filter_datasets_to_teacher_cache(train_data, val_data)
    check_and_record_train_tokens(dataset_tokens(train_data), cfg.output_dir,
                                  is_rank_zero=int(os.environ.get("LOCAL_RANK", "0")) == 0)

    logger.info("Building Datasets")
    if len(train_data) == 0:
        raise RuntimeError("training set is empty (after teacher-cache filtering, if the distillation is on)")
    train_dataloader = DataLoader(
        train_data, **cfg.dataloader.params, shuffle=True, worker_init_fn=_worker_init
    )
    logger.info("Num training samples: %d", len(train_data))
    if len(val_data) == 0:
        val_dataloader = None
        logger.warning("Validation set is empty (teacher-cache filtering); validation is skipped")
    else:
        val_dataloader = DataLoader(
            val_data, **cfg.dataloader.params, shuffle=False, worker_init_fn=_worker_init
        )
        logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer_params = cfg.trainer.params
    # trainer_params['strategy'] = "ddp_find_unused_parameters_true" #TODO
    # TensorBoard always; W&B additionally when cfg.wandb.enable is set.
    # W&B is fail-open telemetry -- see navsim/planning/training/wandb_logging.py.
    callbacks = list(agent.get_training_callbacks())
    ckpt_cfg = cfg.get("checkpoint", None)
    if ckpt_cfg is not None:
        # Explicit, so every epoch is kept. Without this Lightning adds an
        # implicit ModelCheckpoint that retains only the newest epoch.
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
                save_top_k=ckpt_cfg.get("save_top_k", -1),
                save_last=ckpt_cfg.get("save_last", True),
                save_on_train_epoch_end=ckpt_cfg.get(
                    "save_on_train_epoch_end", True
                ),
            )
        )

    trainer = pl.Trainer(
                **trainer_params,
                logger=build_loggers(cfg),
                callbacks=callbacks,
                )

    logger.info("Starting Training")
    resume_checkpoint = cfg.get("resume_checkpoint")
    if resume_checkpoint:
        resume_checkpoint = str(Path(resume_checkpoint).expanduser().resolve())
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(
                f"resume_checkpoint does not exist: {resume_checkpoint}"
            )
        logger.info("Resuming trainer state from %s", resume_checkpoint)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_checkpoint,
    )

if __name__ == "__main__":
    main()
