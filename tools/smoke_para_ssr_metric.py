#!/usr/bin/env python3
"""Exercise metric-aware PARA-SSR on real NAVSIM train/validation scenes.

Example (from an environment with this repository's dependencies installed)::

    python tools/smoke_para_ssr_metric.py \
      --data-root /path/to/navsim/dataset \
      --anchor-path /path/to/train_only_anchors.npz \
      --output-dir /path/to/smoke_output --device cpu

The default uses a small model and two real training plus two real validation
scenes. ``--full-model`` retains the production model dimensions. The tool
generates four genuine metric world caches, executes two Lightning optimizer
steps and validation, and checks strict checkpoint reload with target-free
inference. It does not substitute synthetic labels or use navtest for training.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/dataset")
    parser.add_argument("--anchor-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    anchor_path = args.anchor_path.expanduser().resolve()
    output_dir = (args.output_dir or REPO_ROOT / "work_dirs/smoke_para_ssr_metric" /
                  datetime.now().strftime("%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Train-only anchor archive does not exist: {anchor_path}")
    for folder in ("maps", "navsim_logs/trainval", "sensor_blobs/trainval"):
        if not (data_root / folder).is_dir():
            raise FileNotFoundError(f"Required NAVSIM data directory missing: {data_root / folder}")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() or (output_dir / "smoke.ckpt").exists():
        raise FileExistsError(f"Use a new --output-dir to preserve existing smoke artifacts: {output_dir}")

    # dataclasses.py captures the maps root when imported; set it first.
    os.environ["NUPLAN_MAPS_ROOT"] = str(data_root / "maps")
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["OPENSCENE_DATA_ROOT"] = str(data_root)
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(REPO_ROOT)
    os.environ["NAVSIM_EXP_ROOT"] = str(output_dir)
    sys.path.insert(0, str(REPO_ROOT))

    import csv
    from dataclasses import astuple
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.script.run_training import build_datasets
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    torch.set_num_threads(args.cpu_threads)
    pl.seed_everything(0, workers=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with np.load(anchor_path, allow_pickle=False) as archive:
        anchor_shape = tuple(archive["anchors"].shape)
        anchor_metadata = json.loads(str(archive["metadata_json"].item()))
    if len(anchor_shape) != 3 or anchor_shape[1:] != (8, 3):
        raise ValueError(f"Expected a [K,8,3] anchor archive, got {anchor_shape}")
    if anchor_metadata.get("source_split") != "train":
        raise ValueError("Smoke training requires a vocabulary constructed from train logs only")

    with initialize_config_dir(
        version_base="1.2", config_dir=str(REPO_ROOT / "navsim/planning/script/config/training")
    ):
        cfg = compose(config_name="default_training", overrides=[
            "agent=para_ssr_metric_agent", "scene_filter=navtrain", "split=trainval",
            "scene_filter.max_scenes=2", "experiment_name=smoke_para_ssr_metric",
        ])
    cfg.agent.config.plan_anchor_path = str(anchor_path)
    cfg.agent.config.metric_cache_path = str(output_dir / "metric_cache")
    cfg.agent.config.num_plan_candidates = anchor_shape[0]
    cfg.agent.config.backbone_pretrained = False
    cfg.agent.config.grad_norm_log_interval = 1
    cfg.output_dir = str(output_dir)
    if not args.full_model:
        small = {
            "image_scale": 0.125, "crop_top": 5,
            "bev_h": 8, "bev_w": 16,
            "embed_dims": 32, "num_heads": 4, "ffn_channels": 64,
            "encoder_num_layers": 1, "latent_num_layers": 1,
            "num_query": 8, "max_agents": 8, "det_num_decoder_layers": 1,
            "map_num_vec": 4, "map_max_vec": 4, "map_num_pts_per_vec": 8,
            "map_num_decoder_layers": 1,
        }
        for key, value in small.items():
            OmegaConf.update(cfg.agent.config, key, value, force_add=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    print(f"Building {'full' if args.full_model else 'small'} model; output={output_dir}", flush=True)
    agent = instantiate(cfg.agent)
    train_data, val_data = build_datasets(cfg, agent)
    if len(train_data) != 2 or len(val_data) != 2:
        raise RuntimeError(f"Smoke needs 2 train and 2 validation scenes, got {len(train_data)} and {len(val_data)}")
    train_tokens = train_data._scene_loader.tokens
    val_tokens = val_data._scene_loader.tokens
    if set(train_tokens) & set(val_tokens):
        raise AssertionError("Training and validation tokens overlap")

    world_start = time.perf_counter()
    # An interrupted previous attempt may have left an incomplete cache file.
    # Completed runs are protected above; recompute worlds in unfinished runs.
    processor = MetricCacheProcessor(cfg.agent.config.metric_cache_path, force_feature_computation=True)
    entries = []
    split_logs = {"train": [], "val": []}
    for split, dataset in (("train", train_data), ("val", val_data)):
        for token in dataset._scene_loader.tokens:
            scene = dataset._scene_loader.get_scene_from_token(token)
            split_logs[split].append(scene.scene_metadata.log_name)
            scenario = NavSimScenario(scene, str(data_root / "maps"), "nuplan-maps-v1.0")
            print(f"Generating real metric world cache: {split}/{token}", flush=True)
            entry = processor.compute_metric_cache(scenario)
            if entry is None:
                raise RuntimeError(f"Metric cache generation failed for {split} token {token}")
            entries.append(entry)
    if set(split_logs["train"]) & set(split_logs["val"]):
        raise AssertionError("Training and validation logs overlap")
    metadata_dir = Path(cfg.agent.config.metric_cache_path) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "metric_cache_metadata_node_0.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["file_name"])
        writer.writerows([[str(entry.file_name)] for entry in entries])
    agent.validate_metric_cache((train_data, val_data))
    world_seconds = time.perf_counter() - world_start

    class FiniteChecks(pl.Callback):
        def __init__(self):
            self.train = []
            self.val = []
            self.gradients = []
            self.batch_seconds = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.batch_start = time.perf_counter()

        def on_after_backward(self, trainer, pl_module):
            total_square = 0.0
            group_square = {"metric_head": 0.0, "candidate_cls": 0.0, "ego_fut_decoder": 0.0}
            for name, parameter in pl_module.agent.named_parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"Non-finite gradient in {name}")
                norm_square = float(parameter.grad.detach().float().square().sum())
                total_square += norm_square
                for group in group_square:
                    if group in name:
                        group_square[group] += norm_square
            if total_square <= 0.0 or any(value <= 0 for value in group_square.values()):
                raise AssertionError(f"Missing gradient signal: {group_square}")
            self.gradients.append({"total_norm": total_square ** 0.5,
                                   **{key: value ** 0.5 for key, value in group_square.items()}})

        def _capture(self, destination, pl_module, outputs):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise AssertionError(f"Invalid Lightning scalar loss: {loss}")
            logs = {"lightning_loss": float(loss.detach())}
            for key, value in pl_module.agent.latest_logs.items():
                if not torch.isfinite(value).all():
                    raise AssertionError(f"Non-finite logged value {key}: {value}")
                if value.numel() == 1:
                    logs[key] = float(value.detach())
            if "loss_plan_metric" not in logs or "loss_plan_cls" not in logs:
                raise AssertionError("Metric and candidate classification losses were not executed")
            destination.append(logs)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._capture(self.train, pl_module, outputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.batch_seconds.append(time.perf_counter() - self.batch_start)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self._capture(self.val, pl_module, outputs)

    checks = FiniteChecks()
    module = AgentLightningModule(agent)
    before_metric = agent.para_ssr_model.metric_head.output[-1].weight.detach().clone()
    before_refinement = agent.para_ssr_model.pts_bbox_head.ego_fut_decoder[-1].weight.detach().clone()
    trainer = pl.Trainer(
        fast_dev_run=2, accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1, strategy="auto", precision=32, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[*agent.get_training_callbacks(), checks],
        gradient_clip_val=35.0, gradient_clip_algorithm="norm",
        default_root_dir=str(output_dir),
    )
    fit_start = time.perf_counter()
    trainer.fit(module,
                train_dataloaders=DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False),
                val_dataloaders=DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False))
    fit_seconds = time.perf_counter() - fit_start
    if len(checks.train) != 2 or len(checks.val) != 2 or len(checks.gradients) != 2:
        raise AssertionError("The Lightning smoke did not finish both train/validation batches")
    metric_delta = float((agent.para_ssr_model.metric_head.output[-1].weight.detach().cpu() - before_metric).abs().max())
    refinement_delta = float((agent.para_ssr_model.pts_bbox_head.ego_fut_decoder[-1].weight.detach().cpu() - before_refinement).abs().max())
    if metric_delta <= 0 or refinement_delta <= 0:
        raise AssertionError("Optimizer did not update metric critic and trajectory refinement")

    agent.eval().to(device)
    features, _ = next(iter(DataLoader(val_data, batch_size=1, num_workers=0)))
    features = {key: value.to(device) for key, value in features.items()}
    with torch.no_grad():
        before_reload = agent(features)
    candidates = before_reload["trajectory_candidates"]
    chosen = before_reload["selected_candidate"]
    torch.testing.assert_close(before_reload["trajectory"], candidates[torch.arange(1, device=device), chosen])
    if candidates.shape != (1, anchor_shape[0], 8, 3) or not torch.isfinite(candidates).all():
        raise AssertionError("Invalid generated candidate trajectories")

    # Compare a genuine newly refined candidate against a separately built
    # official evaluator, including its YAML overrides and reference trajectory.
    supervisor = agent._get_metric_supervisor()
    labels = supervisor.score([val_tokens[0]], candidates)
    world = supervisor._get_world(val_tokens[0])
    official_settings = OmegaConf.load(REPO_ROOT / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    official_result = pdm_score(
        world, Trajectory(candidates[0, 0].detach().cpu().double().numpy(), agent._trajectory_sampling),
        instantiate(official_settings.proposal_sampling), instantiate(official_settings.simulator),
        instantiate(official_settings.scorer),
    )
    expected = np.asarray(astuple(official_result))
    np.testing.assert_allclose(labels[0, 0].cpu().numpy(), expected, atol=1e-6, rtol=0.0)

    checkpoint = output_dir / "smoke.ckpt"
    torch.save({"state_dict": module.state_dict()}, checkpoint)
    inference_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=True))
    inference_cfg.config.plan_anchor_path = ""
    inference_cfg.config.metric_cache_path = ""
    inference_cfg.checkpoint_path = str(checkpoint)
    restored = instantiate(inference_cfg)
    restored.initialize()
    restored.eval().to(device)
    if restored._metric_supervisor is not None:
        raise AssertionError("Inference initialized a privileged metric supervisor")
    with torch.no_grad():
        after_reload = restored(features)
    for key in ("trajectory", "trajectory_candidates", "candidate_logits", "metric_logits"):
        torch.testing.assert_close(after_reload[key], before_reload[key], atol=1e-6, rtol=1e-5)
    agent_input = val_data._scene_loader.get_agent_input_from_token(val_tokens[0])
    public_trajectory = restored.compute_trajectory_gpu(agent_input)
    np.testing.assert_allclose(public_trajectory.poses, after_reload["trajectory"][0].cpu().numpy(), atol=1e-6, rtol=1e-5)
    if restored._metric_supervisor is not None:
        raise AssertionError("Target-free inference touched metric world caches")

    summary = {
        "passed": True, "device": str(device), "full_model": args.full_model,
        "data_root": str(data_root), "anchor_path": str(anchor_path), "anchor_metadata": anchor_metadata,
        "train_tokens": train_tokens, "val_tokens": val_tokens, "split_logs": split_logs,
        "model_parameters": sum(parameter.numel() for parameter in agent.parameters()),
        "bev_shape": [cfg.agent.config.bev_h, cfg.agent.config.bev_w],
        "candidate_shape": list(candidates.shape), "metric_label_shape": list(labels.shape),
        "metric_cache_generation_seconds": world_seconds, "lightning_fit_seconds": fit_seconds,
        "train_batch_seconds": checks.batch_seconds,
        "train_logs": checks.train, "val_logs": checks.val, "gradient_checks": checks.gradients,
        "metric_head_max_update": metric_delta, "refinement_head_max_update": refinement_delta,
        "official_metric_parity_max_error": float(np.max(np.abs(labels[0, 0].cpu().numpy() - expected))),
        "strict_checkpoint_reload": True, "inference_without_anchors_or_metric_cache": True,
        "checkpoint_path": str(checkpoint),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None,
    }
    with (output_dir / "summary.json").open("w") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("train_logs", "val_logs", "anchor_metadata")}, indent=2), flush=True)
    print(f"PASS: complete evidence in {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
