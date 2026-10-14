"""Weights & Biases logging for PARA-SSR, alongside TensorBoard.

Ports the intent of the nuScenes ``SSRWandbLoggerHook``
(``projects/mmdet3d_plugin/SSR/hooks/wandb_logger.py``):

* **W&B runs next to TensorBoard, never instead of it.**  Every scalar still
  lands in the event files, so a W&B outage costs telemetry, not the run record.
* **Fail-open.**  W&B is telemetry, not part of the optimisation.  An SDK or
  service failure must never take a distributed training rank down with it, so
  the first exception is logged, W&B is disabled for the rest of the process,
  and training continues.  Set ``non_fatal=false`` only to debug W&B itself.
* **Tag exclusion.**  The nuScenes hook dropped the map head's zero-weight
  bbox/iou terms before upload.  This port has no such terms -- they were
  removed rather than weighted to zero -- so the default exclude set is empty,
  but the mechanism is kept for the same purpose.

Unlike the mmcv hook, nothing here filters what is *optimised* or what reaches
TensorBoard; the exclusion applies to the W&B upload only.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pytorch_lightning.loggers import Logger, TensorBoardLogger

logger = logging.getLogger(__name__)

# The nuScenes hook excluded map.loss_map_{bbox,iou} across the three decoder
# layers. Those terms do not exist in this port, so nothing is dropped by
# default; override ``wandb.exclude`` in the config to add tags.
DEFAULT_EXCLUDE: Sequence[str] = ()


def _matches(tag: str, exclude: Sequence[str]) -> bool:
    """Match a tag with or without its ``train/`` / ``val/`` mode prefix."""
    if tag in exclude:
        return True
    _, _, suffix = tag.partition("/")
    return bool(suffix) and suffix in exclude


def build_wandb_logger(
    output_dir: str,
    project: str = "para-ssr",
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    entity: Optional[str] = None,
    mode: str = "online",
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    non_fatal: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Logger]:
    """Build a fail-open ``WandbLogger``; return ``None`` if unavailable.

    Returning ``None`` rather than raising is deliberate: a missing ``wandb``
    install or a failed login should not block a training run that is otherwise
    fully instrumented through TensorBoard.
    """
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:  # pragma: no cover - depends on install
        logger.warning("W&B logging disabled: %s", exc)
        return None
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning(
            "W&B logging disabled: the 'wandb' package is not installed "
            "(pip install wandb). TensorBoard logging is unaffected."
        )
        return None

    class SafeWandbLogger(WandbLogger):
        """``WandbLogger`` that disables itself instead of raising."""

        def __init__(self, *args, **kwargs):
            self._exclude = tuple(exclude or ())
            self._non_fatal = non_fatal
            self._wandb_disabled = False
            super().__init__(*args, **kwargs)

        def _handle_failure(self, operation: str, exc: BaseException) -> None:
            if not self._non_fatal:
                raise exc
            if not self._wandb_disabled:
                logger.warning(
                    "W&B %s failed (%s: %s); disabling W&B for the rest of this "
                    "process. Training and TensorBoard continue.",
                    operation,
                    type(exc).__name__,
                    exc,
                )
                self._wandb_disabled = True

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            if self._wandb_disabled:
                return
            if self._exclude:
                metrics = {
                    k: v for k, v in metrics.items() if not _matches(k, self._exclude)
                }
            try:
                super().log_metrics(metrics, step)
            except Exception as exc:  # noqa: BLE001 - telemetry must not kill a rank
                self._handle_failure("log_metrics", exc)

        def log_hyperparams(self, *args, **kwargs) -> None:
            if self._wandb_disabled:
                return
            try:
                super().log_hyperparams(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self._handle_failure("log_hyperparams", exc)

        def finalize(self, status: str) -> None:
            if self._wandb_disabled:
                return
            try:
                super().finalize(status)
            except Exception as exc:  # noqa: BLE001
                self._handle_failure("finalize", exc)

    try:
        wandb_logger = SafeWandbLogger(
            project=project,
            name=name,
            group=group,
            tags=list(tags) if tags else None,
            entity=entity,
            mode=mode,
            save_dir=output_dir,
        )
        if config:
            wandb_logger.log_hyperparams(dict(config))
        return wandb_logger
    except Exception as exc:  # noqa: BLE001 - e.g. no API key, no network
        if not non_fatal:
            raise
        logger.warning(
            "W&B logging disabled: could not start a run (%s: %s). "
            "TensorBoard logging is unaffected.",
            type(exc).__name__,
            exc,
        )
        return None


# Config subtrees that distinguish one run from another.  They are uploaded as
# the W&B run config so runs can be filtered/grouped by e.g.
# ``agent.config.plan_det_memory_mode`` without reading the Hydra output dir.
HPARAM_KEYS: Sequence[str] = ("agent", "dataloader", "trainer", "scene_filter", "split")


def _to_plain(node: Any) -> Any:
    """``OmegaConf`` node -> plain Python; falls back to unresolved on bad interpolations."""
    from omegaconf import OmegaConf

    if not OmegaConf.is_config(node):
        return node
    try:
        return OmegaConf.to_container(node, resolve=True)
    except Exception:  # noqa: BLE001 - e.g. ${oc.env:...} unset in this process
        return OmegaConf.to_container(node, resolve=False)


def hparams_from_cfg(cfg) -> Dict[str, Any]:
    """Pick the Hydra subtrees worth storing as the W&B run config."""
    out: Dict[str, Any] = {}
    name = cfg.get("experiment_name", None)
    if name is not None:
        out["experiment_name"] = str(name)
    for key in HPARAM_KEYS:
        node = cfg.get(key, None)
        if node is not None:
            out[key] = _to_plain(node)
    return out


def auto_tags_from_cfg(cfg) -> List[str]:
    """Tags derived from the agent config so ablation runs are tellable apart.

    Covers the knobs the current experiments vary: image backbone, the two
    planner memory options from report 41, and the v2 anchor planner.
    """
    agent = cfg.get("agent", None)
    agent_cfg = agent.get("config", None) if agent is not None else None
    if agent_cfg is None:
        return []
    tags: List[str] = []
    arch = agent_cfg.get("image_architecture", None)
    if arch:
        tags.append(f"backbone:{arch}")
    mode = agent_cfg.get("plan_det_memory_mode", None)
    if mode:
        tags.append(f"det_mem:{mode}")
    segments = agent_cfg.get("plan_map_memory_segments", None)
    if segments is not None:
        tags.append(f"map_seg:{segments}")
    if agent_cfg.get("plan_anchor", False):
        tags.append("v2-anchor")
    return tags


def _merge_tags(*groups: Optional[Sequence[str]]) -> Optional[List[str]]:
    merged: List[str] = []
    for group in groups:
        for tag in group or ():
            tag = str(tag)
            if tag and tag not in merged:
                merged.append(tag)
    return merged or None


def build_loggers(cfg) -> List[Logger]:
    """TensorBoard always; W&B when ``cfg.wandb.enable`` is set.

    The TensorBoard logger reproduces Lightning's own default layout
    (``<output_dir>/lightning_logs/version_N``) so enabling W&B does not move
    any existing event files.  The W&B run additionally receives the Hydra
    config subtrees in :data:`HPARAM_KEYS` as its run config and the tags from
    :func:`auto_tags_from_cfg` on top of ``cfg.wandb.tags``.
    """
    output_dir = str(cfg.output_dir)
    loggers: List[Logger] = [TensorBoardLogger(save_dir=output_dir, name="lightning_logs")]

    wandb_cfg = cfg.get("wandb", None)
    if wandb_cfg is None or not wandb_cfg.get("enable", False):
        return loggers

    wandb_logger = build_wandb_logger(
        output_dir=output_dir,
        project=wandb_cfg.get("project", "para-ssr"),
        name=wandb_cfg.get("name", None) or str(cfg.experiment_name),
        group=wandb_cfg.get("group", None),
        tags=_merge_tags(wandb_cfg.get("tags", None), auto_tags_from_cfg(cfg)),
        entity=wandb_cfg.get("entity", None),
        mode=wandb_cfg.get("mode", "online"),
        exclude=wandb_cfg.get("exclude", DEFAULT_EXCLUDE),
        non_fatal=wandb_cfg.get("non_fatal", True),
        config=hparams_from_cfg(cfg),
    )
    if wandb_logger is not None:
        loggers.append(wandb_logger)
    return loggers
