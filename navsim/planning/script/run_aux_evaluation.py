"""Offline detection/map mAP evaluation for the NAVSIM PARA-SSR port.

The official NAVSIM benchmark evaluates only the ego trajectory.  This runner
therefore publishes explicitly non-official ``NAVSIMAux*`` metrics for the two
parallel training heads.  Inference is checkpoint/config bound, token records
are committed one at a time, and aggregate files are published only after the
entire expected token set has passed validation.

The per-token ``.npz`` files are the resume boundary.  Re-running the same
manifest validates and skips them; changing the checkpoint, training config,
token set, evaluator source, or metric protocol requires a new output folder.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
from importlib import metadata as importlib_metadata
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_targets import (
    DET_CLASS_NAMES,
    MAP_CLASS_NAMES,
    ParaSSRTargetBuilder,
)
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.aux_metrics import (
    AUX_METRIC_PROTOCOL_VERSION,
    decode_detection_predictions,
    decode_map_predictions,
    denormalize_map_ground_truth,
    evaluate_auxiliary_records,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/aux_evaluation"
CONFIG_NAME = "default_aux_evaluation"
REPO_ROOT = Path(__file__).resolve().parents[3]
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
PROTOCOL_TOP_K = 100
PROTOCOL_MAP_POINTS = 20
PROTOCOL_MAP_ORDERS = 20
PROTOCOL_MAP_MIN_LENGTH = 1.0
PROTOCOL_PC_RANGE = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)
PRODUCTION_TOKEN_COUNT = 12_146
PRODUCTION_TOKEN_SHA256 = (
    "19cf783cbae935fce54cc459f05be508cfb546b0d92e7a5a122d0fc0d8bd4419"
)
PRODUCTION_DET_GT_COUNTS = {
    "vehicle": 69_142,
    "pedestrian": 34_653,
    "bicycle": 931,
    "traffic_cone": 20_942,
    "barrier": 8_073,
    "czone_sign": 753,
    "generic_object": 61_214,
}
RECORD_KEYS = {
    "token",
    "manifest_identity_sha256",
    "det_pred_boxes",
    "det_pred_scores",
    "det_pred_labels",
    "det_gt_boxes",
    "det_gt_labels",
    "map_pred_points",
    "map_pred_scores",
    "map_pred_labels",
    "map_gt_points",
    "map_gt_labels",
}
SOURCE_PATHS = (
    "navsim/planning/script/run_aux_evaluation.py",
    "navsim/planning/script/config/aux_evaluation/default_aux_evaluation.yaml",
    "scripts/evaluation/eval_para_ssr_aux.sh",
    "navsim/evaluate/aux_metrics.py",
    "navsim/agents/para_ssr/para_ssr_agent.py",
    "navsim/agents/para_ssr/para_ssr_model.py",
    "navsim/agents/para_ssr/para_ssr_features.py",
    "navsim/agents/para_ssr/para_ssr_targets.py",
    "navsim/agents/para_ssr/para_ssr_loss.py",
    "navsim/agents/para_ssr/configs/default.py",
    "navsim/agents/para_ssr/modules/bevformer.py",
    "navsim/agents/para_ssr/modules/grad_balance.py",
    "navsim/agents/para_ssr/modules/losses.py",
    "navsim/agents/para_ssr/modules/det_motion_head.py",
    "navsim/agents/para_ssr/modules/map_head.py",
    "navsim/agents/para_ssr/modules/ms_deform_attn.py",
    "navsim/agents/para_ssr/modules/planner_head.py",
    "navsim/agents/para_ssr/modules/tokenlearner.py",
    "navsim/agents/para_ssr/modules/transformer_blocks.py",
    "navsim/agents/abstract_agent.py",
    "navsim/common/dataclasses.py",
    "navsim/common/dataloader.py",
    "navsim/planning/training/abstract_feature_target_builder.py",
    "navsim/planning/script/config/common/scene_filter/navtest.yaml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _token_sha256(tokens: Sequence[str]) -> str:
    # The trailing newline is part of the protocol and matches report #09.
    return _sha256_bytes("".join(f"{token}\n" for token in tokens).encode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_create_bytes(path: Path, payload: bytes) -> bool:
    """Create ``path`` exactly once without overwriting a concurrent writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_record(path: Path, record: Mapping[str, np.ndarray]) -> None:
    expected_token = _as_token(record["token"])
    expected_identity = _as_scalar_text(
        record["manifest_identity_sha256"], "manifest_identity_sha256"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.savez_compressed(file, **record)
            file.flush()
            os.fsync(file.fileno())
        # Validate exactly what was serialized before making it a resume point.
        loaded, _ = _load_record(temporary_path, expected_token, expected_identity)
        _validate_record(loaded, expected_token, expected_identity)
        try:
            # Hard-link publication is no-clobber.  Two accidentally duplicated
            # shard workers can never overwrite one another between an
            # existence check and rename.
            os.link(temporary_path, path)
        except FileExistsError:
            existing, _ = _load_record(path, expected_token, expected_identity)
            _validate_record(existing, expected_token, expected_identity)
            differing_keys = [
                key
                for key in sorted(RECORD_KEYS)
                if not np.array_equal(np.asarray(existing[key]), np.asarray(loaded[key]))
            ]
            if differing_keys:
                raise RuntimeError(
                    f"concurrent record collision for {expected_token}: "
                    f"different values in {differing_keys}"
                )
            return
        # Validate the winning pathname, not only the temporary inode name.
        _load_record(path, expected_token, expected_identity)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _as_token(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"serialized token must contain one value, got {array.shape}")
    return str(array.reshape(-1)[0])


def _as_scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"serialized {name} must contain one value, got {array.shape}")
    return str(array.reshape(-1)[0])


def _require_finite(name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_record(
    record: Mapping[str, np.ndarray],
    expected_token: str,
    expected_manifest_identity_sha256: str,
) -> None:
    keys = set(record)
    if keys != RECORD_KEYS:
        raise ValueError(
            f"record {expected_token} has keys {sorted(keys)}, expected {sorted(RECORD_KEYS)}"
        )
    token = _as_token(record["token"])
    if token != expected_token:
        raise ValueError(
            f"record token mismatch: file expects {expected_token!r}, contains {token!r}"
        )
    manifest_identity_sha256 = _as_scalar_text(
        record["manifest_identity_sha256"], "manifest_identity_sha256"
    )
    if manifest_identity_sha256 != expected_manifest_identity_sha256:
        raise ValueError(
            f"{token}: record manifest identity mismatch: "
            f"{manifest_identity_sha256} != {expected_manifest_identity_sha256}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_identity_sha256):
        raise ValueError(f"{token}: invalid manifest identity sha256")

    det_pred_boxes = np.asarray(record["det_pred_boxes"])
    det_pred_scores = np.asarray(record["det_pred_scores"])
    det_pred_labels = np.asarray(record["det_pred_labels"])
    det_gt_boxes = np.asarray(record["det_gt_boxes"])
    det_gt_labels = np.asarray(record["det_gt_labels"])
    map_pred_points = np.asarray(record["map_pred_points"])
    map_pred_scores = np.asarray(record["map_pred_scores"])
    map_pred_labels = np.asarray(record["map_pred_labels"])
    map_gt_points = np.asarray(record["map_gt_points"])
    map_gt_labels = np.asarray(record["map_gt_labels"])

    if det_pred_boxes.ndim != 2 or det_pred_boxes.shape[1:] != (9,):
        raise ValueError(f"{token}: det_pred_boxes has shape {det_pred_boxes.shape}")
    if det_gt_boxes.ndim != 2 or det_gt_boxes.shape[1:] != (9,):
        raise ValueError(f"{token}: det_gt_boxes has shape {det_gt_boxes.shape}")
    if det_pred_scores.shape != (len(det_pred_boxes),):
        raise ValueError(f"{token}: detection score count does not match boxes")
    if det_pred_labels.shape != (len(det_pred_boxes),):
        raise ValueError(f"{token}: detection label count does not match boxes")
    if det_gt_labels.shape != (len(det_gt_boxes),):
        raise ValueError(f"{token}: detection GT label count does not match boxes")
    if map_pred_points.ndim != 3 or map_pred_points.shape[-1] != 2:
        raise ValueError(f"{token}: map_pred_points has shape {map_pred_points.shape}")
    if map_gt_points.ndim != 3 or map_gt_points.shape[-1] != 2:
        raise ValueError(f"{token}: map_gt_points has shape {map_gt_points.shape}")
    if map_pred_scores.shape != (len(map_pred_points),):
        raise ValueError(f"{token}: map score count does not match polylines")
    if map_pred_labels.shape != (len(map_pred_points),):
        raise ValueError(f"{token}: map label count does not match polylines")
    if map_gt_labels.shape != (len(map_gt_points),):
        raise ValueError(f"{token}: map GT label count does not match polylines")
    if len(det_pred_boxes) != PROTOCOL_TOP_K:
        raise ValueError(
            f"{token}: protocol requires exactly {PROTOCOL_TOP_K} detection "
            f"predictions, got {len(det_pred_boxes)}"
        )
    if len(map_pred_points) != PROTOCOL_TOP_K:
        raise ValueError(
            f"{token}: protocol requires exactly {PROTOCOL_TOP_K} map "
            f"predictions, got {len(map_pred_points)}"
        )
    if map_pred_points.shape[1] != PROTOCOL_MAP_POINTS:
        raise ValueError(
            f"{token}: map predictions must have {PROTOCOL_MAP_POINTS} raw points, "
            f"got {map_pred_points.shape[1]}"
        )
    if map_gt_points.shape[1] != PROTOCOL_MAP_POINTS:
        raise ValueError(
            f"{token}: map GT must have {PROTOCOL_MAP_POINTS} raw points, "
            f"got {map_gt_points.shape[1]}"
        )

    for name, value in record.items():
        if name not in {"token", "manifest_identity_sha256"}:
            _require_finite(f"{token}:{name}", np.asarray(value))
    if len(det_pred_boxes) and np.any(det_pred_boxes[:, 3:6] <= 0):
        raise ValueError(f"{token}: predicted detection box has non-positive size")
    if len(det_gt_boxes) and np.any(det_gt_boxes[:, 3:6] <= 0):
        raise ValueError(f"{token}: GT detection box has non-positive size")
    det_lower = np.asarray(PROTOCOL_PC_RANGE[:3], dtype=np.float32)
    det_upper = np.asarray(PROTOCOL_PC_RANGE[3:6], dtype=np.float32)
    if len(det_pred_boxes) and np.any(
        (det_pred_boxes[:, :3] < det_lower - 1e-5)
        | (det_pred_boxes[:, :3] > det_upper + 1e-5)
    ):
        raise ValueError(f"{token}: predicted detection center is outside pc_range")
    if len(det_gt_boxes) and np.any(
        (det_gt_boxes[:, :2] < det_lower[:2] - 1e-5)
        | (det_gt_boxes[:, :2] > det_upper[:2] + 1e-5)
    ):
        raise ValueError(f"{token}: GT detection center is outside x/y pc_range")
    if np.any((det_pred_scores < 0.0) | (det_pred_scores > 1.0)):
        raise ValueError(f"{token}: detection scores must be in [0, 1]")
    if np.any((map_pred_scores < 0.0) | (map_pred_scores > 1.0)):
        raise ValueError(f"{token}: map scores must be in [0, 1]")
    for name, labels in (
        ("det_pred_labels", det_pred_labels),
        ("det_gt_labels", det_gt_labels),
        ("map_pred_labels", map_pred_labels),
        ("map_gt_labels", map_gt_labels),
    ):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{token}: {name} must use an integer dtype")
    if np.any((det_pred_labels < 0) | (det_pred_labels >= len(DET_CLASS_NAMES))):
        raise ValueError(f"{token}: prediction detection label is out of range")
    if np.any((det_gt_labels < 0) | (det_gt_labels >= len(DET_CLASS_NAMES))):
        raise ValueError(f"{token}: GT detection label is out of range")
    if np.any((map_pred_labels < 0) | (map_pred_labels >= len(MAP_CLASS_NAMES))):
        raise ValueError(f"{token}: prediction map label is out of range")
    if np.any((map_gt_labels < 0) | (map_gt_labels >= len(MAP_CLASS_NAMES))):
        raise ValueError(f"{token}: GT map label is out of range")


def _load_record(
    path: Path, expected_token: str, expected_manifest_identity_sha256: str
) -> Tuple[Dict[str, np.ndarray], str]:
    payload = path.read_bytes()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            record = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise RuntimeError(f"cannot read auxiliary record {path}") from exc
    _validate_record(record, expected_token, expected_manifest_identity_sha256)
    return record, _sha256_bytes(payload)


class _AuxiliaryTokenDataset(TorchDataset):
    """Build model features and uncapped evaluation GT from one scene load.

    The ordinary training dataset is intentionally not used here: its detection
    and map targets are padded/capped for batching.  Evaluation GT is variable
    length and must remain complete, so the collate function below stacks only
    features and retains one target dictionary per token.
    """

    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: Sequence[AbstractFeatureBuilder],
        evaluation_target_builder: ParaSSRTargetBuilder,
        tokens: Sequence[str],
    ):
        self._scene_loader = scene_loader
        self._feature_builders = list(feature_builders)
        self._evaluation_target_builder = evaluation_target_builder
        loader_tokens = list(scene_loader.tokens)
        if len(set(loader_tokens)) != len(loader_tokens):
            raise RuntimeError("SceneLoader contains duplicate tokens")
        loader_token_set = set(loader_tokens)
        missing = [token for token in tokens if token not in loader_token_set]
        if missing:
            raise RuntimeError(f"dataset is missing token {missing[0]}")
        self._tokens = list(tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int):
        token = self._tokens[index]
        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(builder.compute_features(agent_input))
        targets = {
            **self._evaluation_target_builder.compute_detection_evaluation_targets(
                scene
            ),
            **self._evaluation_target_builder.compute_map_evaluation_targets(scene),
        }
        return token, features, targets


def _collate_auxiliary_batch(batch):
    tokens, features, variable_targets = zip(*batch)
    return list(tokens), default_collate(list(features)), list(variable_targets)


def _validate_token_list(tokens: Iterable[str], name: str) -> List[str]:
    values = [str(token) for token in tokens]
    if not values:
        raise ValueError(f"{name} token list is empty")
    invalid = [token for token in values if not TOKEN_PATTERN.fullmatch(token)]
    if invalid:
        raise ValueError(f"{name} contains unsafe token {invalid[0]!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate tokens")
    return sorted(values)


def _source_hashes() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required evaluator source does not exist: {path}")
        result[relative] = _sha256_file(path)
    return result


def _build_identity(
    cfg: DictConfig,
    checkpoint_path: Path,
    training_config_path: Path,
    tokens: Sequence[str],
) -> Dict[str, object]:
    return {
        "protocol": "NAVSIM_PARASSR_AUX_MAP_V1",
        "metric_protocol_version": int(AUX_METRIC_PROTOCOL_VERSION),
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
        },
        "training_config": {
            "path": str(training_config_path),
            "bytes": training_config_path.stat().st_size,
            "sha256": _sha256_file(training_config_path),
        },
        "dataset": {
            "split": str(cfg.split),
            "scene_filter": str(cfg.scene_filter_name),
            "navsim_log_path": str(Path(cfg.navsim_log_path).resolve()),
            "sensor_blobs_path": str(Path(cfg.sensor_blobs_path).resolve()),
            "nuplan_maps_root": str(
                Path(os.environ["NUPLAN_MAPS_ROOT"]).expanduser().resolve()
            ),
            "nuplan_map_version": "nuplan-maps-v1.0",
            "num_tokens": len(tokens),
            "token_sha256": _token_sha256(tokens),
        },
        "inference": {
            "run_aux": True,
            "precision": "fp32",
            "det_max_predictions": int(cfg.det_max_predictions),
            "map_max_predictions": int(cfg.map_max_predictions),
            "map_raw_points_per_line": PROTOCOL_MAP_POINTS,
            "map_equivalent_orders": PROTOCOL_MAP_ORDERS,
            "map_min_length_m": PROTOCOL_MAP_MIN_LENGTH,
            "pc_range": list(PROTOCOL_PC_RANGE),
            "batch_size": int(cfg.dataloader.batch_size),
            "checkpoint_only_overrides": {
                "test_aux_heads": True,
                "backbone_pretrained": False,
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
            "numpy": str(np.__version__),
            "timm": importlib_metadata.version("timm"),
            "opencv_python": importlib_metadata.version("opencv-python"),
            "shapely": importlib_metadata.version("shapely"),
            "nuplan_devkit": importlib_metadata.version("nuplan-devkit"),
            "hydra_core": importlib_metadata.version("hydra-core"),
            "omegaconf": importlib_metadata.version("omegaconf"),
        },
        "sources": _source_hashes(),
    }


def _write_or_validate_manifest(path: Path, identity: Dict[str, object]) -> str:
    identity_sha256 = _sha256_bytes(_canonical_json_bytes(identity))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(
                f"output directory belongs to another auxiliary evaluation: {path}"
            )
        if existing.get("identity") != identity:
            raise RuntimeError(f"manifest identity payload is inconsistent: {path}")
        return identity_sha256
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "identity_sha256": identity_sha256,
        "identity": identity,
    }
    _atomic_create_bytes(path, _canonical_json_bytes(manifest))
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("identity_sha256") != identity_sha256:
        raise RuntimeError(
            f"concurrent process created a different auxiliary manifest: {path}"
        )
    if existing.get("identity") != identity:
        raise RuntimeError(f"concurrent manifest identity is inconsistent: {path}")
    return identity_sha256


def _load_training_agent_config(
    training_config_path: Path, checkpoint_path: Path
) -> DictConfig:
    training_config = OmegaConf.load(training_config_path)
    if "agent" not in training_config:
        raise KeyError(f"archived training config has no agent section: {training_config_path}")
    agent_config = OmegaConf.create(
        OmegaConf.to_container(training_config.agent, resolve=False)
    )
    OmegaConf.set_struct(agent_config, False)
    agent_config.checkpoint_path = str(checkpoint_path)
    agent_config.resume_from_checkpoint = False
    agent_config.config.test_aux_heads = True
    # Strict checkpoint loading replaces every model tensor.  Avoid an
    # unnecessary timm download/cache dependency during held-out evaluation.
    agent_config.config.backbone_pretrained = False
    return agent_config


def _build_agent(
    training_config_path: Path, checkpoint_path: Path, device: torch.device
) -> ParaSSRAgent:
    agent = instantiate(_load_training_agent_config(training_config_path, checkpoint_path))
    if not isinstance(agent, ParaSSRAgent):
        raise TypeError(f"archived config instantiated {type(agent).__name__}, not ParaSSRAgent")
    if not bool(agent.config.use_det_motion_head) or not bool(agent.config.use_map_head):
        raise RuntimeError("auxiliary evaluation requires both detection and map heads")
    if int(agent.config.map_num_pts_per_vec) != PROTOCOL_MAP_POINTS:
        raise RuntimeError(
            "metric protocol V1 requires archived map_num_pts_per_vec="
            f"{PROTOCOL_MAP_POINTS}, got {agent.config.map_num_pts_per_vec}"
        )
    if int(agent.config.map_num_orders) != PROTOCOL_MAP_ORDERS:
        raise RuntimeError(
            f"metric protocol V1 fixes map_num_orders={PROTOCOL_MAP_ORDERS}, "
            f"got {agent.config.map_num_orders}"
        )
    if float(agent.config.map_min_length) != PROTOCOL_MAP_MIN_LENGTH:
        raise RuntimeError(
            f"metric protocol V1 fixes map_min_length={PROTOCOL_MAP_MIN_LENGTH}, "
            f"got {agent.config.map_min_length}"
        )
    if tuple(float(value) for value in agent.config.pc_range) != PROTOCOL_PC_RANGE:
        raise RuntimeError(
            f"metric protocol V1 fixes pc_range={PROTOCOL_PC_RANGE}, "
            f"got {tuple(agent.config.pc_range)}"
        )
    if int(agent.config.num_query) * int(agent.config.num_det_classes) < PROTOCOL_TOP_K:
        raise RuntimeError("detection head has fewer logits than protocol top-100")
    if int(agent.config.map_num_vec) * int(agent.config.map_num_classes) < PROTOCOL_TOP_K:
        raise RuntimeError("map head has fewer logits than protocol top-100")
    agent.initialize()
    agent.float().to(device)
    agent.eval()
    floating_dtypes = {
        tensor.dtype
        for tensor in list(agent.parameters()) + list(agent.buffers())
        if tensor.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise RuntimeError(f"model is not fully fp32 after loading: {floating_dtypes}")
    return agent


def _validate_protocol_options(cfg: DictConfig) -> None:
    if int(cfg.det_max_predictions) != PROTOCOL_TOP_K:
        raise ValueError(
            f"metric protocol V1 fixes det_max_predictions={PROTOCOL_TOP_K}"
        )
    if int(cfg.map_max_predictions) != PROTOCOL_TOP_K:
        raise ValueError(
            f"metric protocol V1 fixes map_max_predictions={PROTOCOL_TOP_K}"
        )
    if isinstance(cfg.num_shards, bool) or int(cfg.num_shards) < 1:
        raise ValueError(f"num_shards must be a positive integer, got {cfg.num_shards}")
    if isinstance(cfg.shard_index, bool) or not (
        0 <= int(cfg.shard_index) < int(cfg.num_shards)
    ):
        raise ValueError(
            f"shard_index must be in [0, {int(cfg.num_shards) - 1}], "
            f"got {cfg.shard_index}"
        )
    if bool(cfg.extract_only) and bool(cfg.aggregate_only):
        raise ValueError("extract_only and aggregate_only are mutually exclusive")
    if int(cfg.num_shards) > 1 and not bool(cfg.extract_only):
        raise ValueError("num_shards > 1 is only valid with extract_only=true")
    if bool(cfg.aggregate_only) and (
        int(cfg.num_shards) != 1 or int(cfg.shard_index) != 0
    ):
        raise ValueError("aggregate_only requires num_shards=1 and shard_index=0")
    if int(cfg.dataloader.batch_size) < 1:
        raise ValueError("dataloader.batch_size must be positive")
    if int(cfg.dataloader.num_workers) < 0:
        raise ValueError("dataloader.num_workers must be non-negative")


def _record_from_batch_item(
    token: str,
    decoded_detection: Mapping[str, np.ndarray],
    decoded_map: Mapping[str, np.ndarray],
    targets: Mapping[str, torch.Tensor],
    pc_range: Sequence[float],
    manifest_identity_sha256: str,
) -> Dict[str, np.ndarray]:
    det_gt_boxes = targets["gt_boxes"].detach().cpu().numpy()
    det_gt_labels = targets["gt_labels"].detach().cpu().numpy()
    normalized_map_gt = targets["gt_map_pts"].detach().cpu().numpy()
    map_gt_points = denormalize_map_ground_truth(normalized_map_gt, pc_range)
    map_gt_labels = targets["gt_map_labels"].detach().cpu().numpy()
    record = {
        "token": np.asarray(token),
        "manifest_identity_sha256": np.asarray(manifest_identity_sha256),
        "det_pred_boxes": np.asarray(decoded_detection["boxes"], dtype=np.float32),
        "det_pred_scores": np.asarray(decoded_detection["scores"], dtype=np.float32),
        "det_pred_labels": np.asarray(decoded_detection["labels"], dtype=np.int64),
        "det_gt_boxes": np.asarray(det_gt_boxes, dtype=np.float32),
        "det_gt_labels": np.asarray(det_gt_labels, dtype=np.int64),
        "map_pred_points": np.asarray(decoded_map["points"], dtype=np.float32),
        "map_pred_scores": np.asarray(decoded_map["scores"], dtype=np.float32),
        "map_pred_labels": np.asarray(decoded_map["labels"], dtype=np.int64),
        "map_gt_points": np.asarray(map_gt_points, dtype=np.float32),
        "map_gt_labels": np.asarray(map_gt_labels, dtype=np.int64),
    }
    _validate_record(record, token, manifest_identity_sha256)
    return record


def _run_missing_inference(
    cfg: DictConfig,
    missing_tokens: Sequence[str],
    checkpoint_path: Path,
    training_config_path: Path,
    scene_filter,
    records_dir: Path,
    manifest_identity_sha256: str,
) -> None:
    if not missing_tokens:
        return
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    agent = _build_agent(training_config_path, checkpoint_path, device)

    inference_scene_filter = replace(
        scene_filter, tokens=list(missing_tokens), max_scenes=None
    )
    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=inference_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    target_builders = agent.get_target_builders()
    if len(target_builders) != 1 or not isinstance(
        target_builders[0], ParaSSRTargetBuilder
    ):
        raise TypeError(
            "PARA-SSR auxiliary evaluation requires exactly one "
            "ParaSSRTargetBuilder"
        )
    dataset = _AuxiliaryTokenDataset(
        scene_loader,
        agent.get_feature_builders(),
        target_builders[0],
        missing_tokens,
    )
    loader_kwargs = {
        "batch_size": int(cfg.dataloader.batch_size),
        "num_workers": int(cfg.dataloader.num_workers),
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "shuffle": False,
        "drop_last": False,
        "collate_fn": _collate_auxiliary_batch,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.dataloader.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(cfg.dataloader.persistent_workers)
    dataloader = DataLoader(dataset, **loader_kwargs)

    model = agent.para_ssr_model
    pc_range = tuple(float(value) for value in agent.config.pc_range)
    progress = tqdm(total=len(missing_tokens), desc="Auxiliary inference", unit="token")
    with torch.inference_mode():
        for batch_tokens, features, targets in dataloader:
            ordered_batch_tokens = [str(token) for token in batch_tokens]
            features_gpu = {}
            for key, value in features.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"feature {key!r} is not a tensor")
                target_dtype = torch.float32 if value.is_floating_point() else value.dtype
                features_gpu[key] = value.to(
                    device=device, dtype=target_dtype, non_blocking=True
                )
            # No autocast: deformable attention and all heads stay in fp32.
            predictions = model(features_gpu, run_aux=True)
            required = {
                "all_cls_scores",
                "all_bbox_preds",
                "all_map_cls_scores",
                "all_map_pts_preds",
            }
            missing_outputs = required - set(predictions)
            if missing_outputs:
                raise RuntimeError(
                    f"run_aux=True did not emit {sorted(missing_outputs)}"
                )
            non_fp32_outputs = {
                name: predictions[name].dtype
                for name in required
                if predictions[name].dtype != torch.float32
            }
            if non_fp32_outputs:
                raise RuntimeError(
                    f"auxiliary inference did not stay fp32: {non_fp32_outputs}"
                )
            decoded_detection = decode_detection_predictions(
                predictions, max_predictions=int(cfg.det_max_predictions)
            )
            decoded_map = decode_map_predictions(
                predictions,
                pc_range=pc_range,
                max_predictions=int(cfg.map_max_predictions),
            )
            if not (
                len(ordered_batch_tokens)
                == len(decoded_detection)
                == len(decoded_map)
            ):
                raise RuntimeError("auxiliary decoder batch size mismatch")
            for batch_index, token in enumerate(ordered_batch_tokens):
                record = _record_from_batch_item(
                    token,
                    decoded_detection[batch_index],
                    decoded_map[batch_index],
                    targets[batch_index],
                    pc_range,
                    manifest_identity_sha256,
                )
                _atomic_write_record(records_dir / f"{token}.npz", record)
                progress.update(1)
    progress.close()


def _validate_record_inventory(
    records_dir: Path,
    expected_tokens: Sequence[str],
    manifest_identity_sha256: str,
) -> Tuple[List[str], List[str]]:
    expected = set(expected_tokens)
    present_paths = list(records_dir.glob("*.npz")) if records_dir.is_dir() else []
    present_tokens = [path.stem for path in present_paths]
    if len(present_tokens) != len(set(present_tokens)):
        raise RuntimeError("duplicate record filenames were found")
    extras = sorted(set(present_tokens) - expected)
    if extras:
        raise RuntimeError(f"record directory contains unexpected token {extras[0]}")
    valid = []
    for token in sorted(set(present_tokens)):
        _load_record(
            records_dir / f"{token}.npz", token, manifest_identity_sha256
        )
        valid.append(token)
    missing = sorted(expected - set(valid))
    return valid, missing


def _metrics_csv_bytes(metrics: Mapping[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    fields = (
        "task",
        "class",
        "threshold_m",
        "AP",
        "max_recall",
        "num_gt",
        "num_predictions",
        "class_AP",
        "task_mAP",
    )
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for task_name in ("detection", "map"):
        task = metrics[task_name]
        for class_name, class_result in task["classes"].items():
            for threshold, threshold_result in class_result["thresholds"].items():
                writer.writerow(
                    {
                        "task": task_name,
                        "class": class_name,
                        "threshold_m": threshold,
                        "AP": threshold_result["AP"],
                        "max_recall": threshold_result["max_recall"],
                        "num_gt": class_result["num_gt"],
                        "num_predictions": class_result["num_predictions"],
                        "class_AP": class_result["AP"],
                        "task_mAP": task["mAP"],
                    }
                )
    return stream.getvalue().encode("utf-8")


def _aggregate_records(
    records_dir: Path,
    expected_tokens: Sequence[str],
    manifest_identity_sha256: str,
) -> Tuple[Dict[str, object], str]:
    records_digest = hashlib.sha256()

    def records() -> Iterator[Mapping[str, np.ndarray]]:
        for token in expected_tokens:
            record, file_sha256 = _load_record(
                records_dir / f"{token}.npz", token, manifest_identity_sha256
            )
            records_digest.update(f"{token} {file_sha256}\n".encode("utf-8"))
            yield record

    metrics = evaluate_auxiliary_records(records())
    if int(metrics.get("num_tokens", -1)) != len(expected_tokens):
        raise RuntimeError(
            f"aggregator consumed {metrics.get('num_tokens')} tokens, expected "
            f"{len(expected_tokens)}"
        )
    return metrics, records_digest.hexdigest()


def _publish_final_results(
    output_dir: Path,
    manifest_identity_sha256: str,
    identity: Mapping[str, object],
    metrics: Dict[str, object],
    records_sha256: str,
) -> None:
    csv_path = output_dir / "aux_metrics.csv"
    json_path = output_dir / "aux_metrics.json"
    csv_payload = _metrics_csv_bytes(metrics)
    csv_sha256 = _sha256_bytes(csv_payload)

    def validate_receipt(receipt: Mapping[str, object]) -> None:
        required_keys = {
            "schema_version",
            "completed_at_utc",
            "manifest_identity_sha256",
            "checkpoint_sha256",
            "training_config_sha256",
            "token_sha256",
            "num_tokens",
            "artifacts",
            "metrics",
        }
        if set(receipt) != required_keys or receipt.get("schema_version") != 1:
            raise RuntimeError("final JSON has an invalid schema")
        expected_scalars = {
            "manifest_identity_sha256": manifest_identity_sha256,
            "checkpoint_sha256": identity["checkpoint"]["sha256"],
            "training_config_sha256": identity["training_config"]["sha256"],
            "token_sha256": identity["dataset"]["token_sha256"],
            "num_tokens": identity["dataset"]["num_tokens"],
        }
        for key, expected in expected_scalars.items():
            if receipt.get(key) != expected:
                raise RuntimeError(f"final JSON {key} does not match its manifest")
        if receipt.get("metrics") != metrics:
            raise RuntimeError("existing final JSON does not match recomputed metrics")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "records_sha256",
            "csv",
            "csv_sha256",
        }:
            raise RuntimeError("final JSON has an invalid artifact receipt")
        expected_artifacts = {
            "records_sha256": records_sha256,
            "csv": str(csv_path),
            "csv_sha256": csv_sha256,
        }
        if dict(artifacts) != expected_artifacts:
            raise RuntimeError("final JSON artifact receipt does not match current files")
        completed_at = receipt.get("completed_at_utc")
        if not isinstance(completed_at, str) or not completed_at.endswith("Z"):
            raise RuntimeError("final JSON has an invalid completion timestamp")

    if json_path.exists():
        if not csv_path.is_file():
            raise RuntimeError(f"completion JSON exists without its CSV: {json_path}")
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        validate_receipt(existing)
        if existing["artifacts"]["csv_sha256"] != _sha256_file(csv_path):
            raise RuntimeError("existing final CSV hash does not match its JSON receipt")
        if csv_path.read_bytes() != csv_payload:
            raise RuntimeError("existing final CSV does not match recomputed metrics")
        logger.info("Existing complete auxiliary evaluation verified: %s", json_path)
        return

    # CSV is staged first.  The JSON is the final completion marker and binds
    # its exact bytes; an interruption between the two is safely resumable.
    _atomic_write_bytes(csv_path, csv_payload)
    result = {
        "schema_version": 1,
        "completed_at_utc": _utc_now(),
        "manifest_identity_sha256": manifest_identity_sha256,
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "training_config_sha256": identity["training_config"]["sha256"],
        "token_sha256": identity["dataset"]["token_sha256"],
        "num_tokens": identity["dataset"]["num_tokens"],
        "artifacts": {
            "records_sha256": records_sha256,
            "csv": str(csv_path),
            "csv_sha256": csv_sha256,
        },
        "metrics": metrics,
    }
    _atomic_write_bytes(json_path, _canonical_json_bytes(result))
    # Read back both completion artifacts before reporting success.
    published = json.loads(json_path.read_text(encoding="utf-8"))
    if published != result:
        raise RuntimeError("published auxiliary JSON failed exact readback")
    validate_receipt(published)
    if published["artifacts"]["csv_sha256"] != _sha256_file(csv_path):
        raise RuntimeError("published auxiliary CSV failed its completion hash")


def _validate_production_metrics(metrics: Mapping[str, object]) -> None:
    for task_name, class_names in (
        ("detection", DET_CLASS_NAMES),
        ("map", MAP_CLASS_NAMES),
    ):
        task = metrics.get(task_name)
        if not isinstance(task, Mapping):
            raise RuntimeError(f"production metrics are missing {task_name}")
        task_map = task.get("mAP")
        if task_map is None or not np.isfinite(float(task_map)):
            raise RuntimeError(f"production {task_name} mAP is not finite")
        classes = task.get("classes")
        if not isinstance(classes, Mapping) or set(classes) != set(class_names):
            raise RuntimeError(f"production {task_name} class set is invalid")
        for class_name in class_names:
            num_gt = int(classes[class_name].get("num_gt", 0))
            if num_gt <= 0:
                raise RuntimeError(
                    f"production {task_name}/{class_name} has no ground truth"
                )
    actual_det_counts = {
        class_name: int(metrics["detection"]["classes"][class_name]["num_gt"])
        for class_name in DET_CLASS_NAMES
    }
    if actual_det_counts != PRODUCTION_DET_GT_COUNTS:
        raise RuntimeError(
            "production detection GT counts do not match the uncapped navtest "
            f"reference: actual={actual_det_counts}, expected={PRODUCTION_DET_GT_COUNTS}"
        )
    scenes_over_cap = int(metrics["map"].get("scenes_over_training_gt_cap_100", 0))
    if scenes_over_cap <= 0:
        raise RuntimeError(
            "production map GT never exceeded the training cap; uncapped "
            "evaluation-target path is not proven active"
        )


def _production_guard(
    cfg: DictConfig, configured_tokens: Sequence[str], scene_filter
) -> None:
    if not bool(cfg.production_guard):
        return
    errors = []
    if str(cfg.split) != "test":
        errors.append(f"split={cfg.split!s}, expected test")
    if str(cfg.scene_filter_name) != "navtest":
        errors.append(f"scene_filter_name={cfg.scene_filter_name!s}, expected navtest")
    scene_contract = {
        "num_history_frames": (scene_filter.num_history_frames, 4),
        "num_future_frames": (scene_filter.num_future_frames, 10),
        "frame_interval": (scene_filter.frame_interval, 1),
        "has_route": (scene_filter.has_route, True),
        "max_scenes": (scene_filter.max_scenes, None),
    }
    for name, (actual, expected) in scene_contract.items():
        if actual != expected:
            errors.append(f"scene_filter.{name}={actual!r}, expected {expected!r}")
    if len(configured_tokens) != PRODUCTION_TOKEN_COUNT:
        errors.append(
            f"configured tokens={len(configured_tokens)}, expected {PRODUCTION_TOKEN_COUNT}"
        )
    token_hash = _token_sha256(configured_tokens)
    if token_hash != PRODUCTION_TOKEN_SHA256:
        errors.append(
            f"token_sha256={token_hash}, expected {PRODUCTION_TOKEN_SHA256}"
        )
    if errors:
        raise RuntimeError("production auxiliary-eval guard failed: " + "; ".join(errors))


@hydra.main(version_base="1.2", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    _validate_protocol_options(cfg)
    checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
    training_config_path = Path(cfg.training_config_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"archived training config does not exist: {training_config_path}"
        )

    scene_filter = instantiate(cfg.scene_filter)
    if scene_filter.tokens is None:
        raise RuntimeError("auxiliary evaluation requires an explicit token allow-list")
    configured_tokens = _validate_token_list(scene_filter.tokens, "scene_filter")
    _production_guard(cfg, configured_tokens, scene_filter)

    if bool(cfg.dry_run):
        identity = _build_identity(
            cfg, checkpoint_path, training_config_path, configured_tokens
        )
        logger.info(
            "Dry run passed: tokens=%d token_sha256=%s checkpoint_sha256=%s",
            len(configured_tokens),
            identity["dataset"]["token_sha256"],
            identity["checkpoint"]["sha256"],
        )
        return

    # Filtering without sensors is enough to validate the exact held-out set
    # before allocating a model or touching the record directory.
    validation_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    actual_tokens = _validate_token_list(validation_loader.tokens, "SceneLoader")
    if actual_tokens != configured_tokens:
        missing = sorted(set(configured_tokens) - set(actual_tokens))
        extra = sorted(set(actual_tokens) - set(configured_tokens))
        raise RuntimeError(
            "SceneLoader token set does not equal the configured allow-list: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:1]}, first_extra={extra[:1]}"
        )
    if bool(cfg.production_guard) and len(actual_tokens) != PRODUCTION_TOKEN_COUNT:
        raise RuntimeError(
            f"production SceneLoader returned {len(actual_tokens)} tokens, expected "
            f"{PRODUCTION_TOKEN_COUNT}"
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    identity = _build_identity(
        cfg, checkpoint_path, training_config_path, actual_tokens
    )
    manifest_path = output_dir / "manifest.json"
    manifest_identity_sha256 = _write_or_validate_manifest(manifest_path, identity)

    valid_tokens, missing_tokens = _validate_record_inventory(
        records_dir, actual_tokens, manifest_identity_sha256
    )
    logger.info(
        "Auxiliary record inventory: valid=%d missing=%d expected=%d",
        len(valid_tokens),
        len(missing_tokens),
        len(actual_tokens),
    )
    final_json = output_dir / "aux_metrics.json"
    if final_json.exists() and missing_tokens:
        raise RuntimeError(
            "a final result exists but its per-token record set is incomplete; "
            "refusing to mix completed and partial state"
        )

    assigned_tokens = actual_tokens[
        int(cfg.shard_index) :: int(cfg.num_shards)
    ]
    missing_set = set(missing_tokens)
    assigned_missing_tokens = [
        token for token in assigned_tokens if token in missing_set
    ]
    logger.info(
        "Shard assignment: shard=%d/%d assigned=%d missing=%d mode=%s",
        int(cfg.shard_index),
        int(cfg.num_shards),
        len(assigned_tokens),
        len(assigned_missing_tokens),
        "aggregate_only"
        if bool(cfg.aggregate_only)
        else ("extract_only" if bool(cfg.extract_only) else "extract_and_aggregate"),
    )
    if not bool(cfg.aggregate_only):
        _run_missing_inference(
            cfg,
            assigned_missing_tokens,
            checkpoint_path,
            training_config_path,
            scene_filter,
            records_dir,
            manifest_identity_sha256,
        )

    valid_tokens, missing_tokens = _validate_record_inventory(
        records_dir, actual_tokens, manifest_identity_sha256
    )
    if bool(cfg.extract_only):
        missing_after_set = set(missing_tokens)
        shard_missing_after = [
            token for token in assigned_tokens if token in missing_after_set
        ]
        if shard_missing_after:
            raise RuntimeError(
                f"shard extraction is incomplete: missing={len(shard_missing_after)}, "
                f"first_missing={shard_missing_after[:1]}"
            )
        logger.info(
            "Shard extraction complete; aggregate files intentionally not published: "
            "shard=%d/%d records=%d global_valid=%d/%d",
            int(cfg.shard_index),
            int(cfg.num_shards),
            len(assigned_tokens),
            len(valid_tokens),
            len(actual_tokens),
        )
        return
    if missing_tokens or valid_tokens != actual_tokens:
        raise RuntimeError(
            "auxiliary evaluation is partial; no final result will be published: "
            f"valid={len(valid_tokens)}, expected={len(actual_tokens)}, "
            f"first_missing={missing_tokens[:1]}"
        )
    metrics, records_sha256 = _aggregate_records(
        records_dir, actual_tokens, manifest_identity_sha256
    )
    if bool(cfg.production_guard):
        _validate_production_metrics(metrics)
    _publish_final_results(
        output_dir,
        manifest_identity_sha256,
        identity,
        metrics,
        records_sha256,
    )
    logger.info(
        "Auxiliary evaluation complete: det mAP=%.6f map mAP=%.6f output=%s",
        metrics["detection"]["mAP"],
        metrics["map"]["mAP"],
        output_dir,
    )


if __name__ == "__main__":
    main()
