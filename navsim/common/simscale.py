"""SimScale synthetic scenes (OpenDriveLab/SimScale, planner-based pseudo-expert rounds) for training.

Layout under SSR_SIMSCALE_ROOT (navsim/common/ssr_paths.py), as the SimScale archives unpack:
    synthetic_reaction_pdm_v1.0-<r>/meta_datas/<log>-<scene>-<rrr>.pkl     one scene per pickle
    synthetic_reaction_pdm_v1.0-<r>/sensor_blobs_hist/<scene>/CAM_*/<token>.jpg

A scene pickle is a 12-frame OpenScene log at 2 Hz: frames 0-3 history (images in sensor_blobs_hist), frame 3 the
current frame, frames 4-11 the 4 s pseudo-expert future (no images; sensor_blobs_fut is not used).  Keys are the real
log keys, except lidar_path None (no LiDAR) and 4x4 pose matrices with a zero bottom row (navsim builds poses from
translation + quaternion, so those are never read).  The scene token is frames[3]["token"], '<16 hex>-<rrr>': the
16 hex part is the real token at the same time stamp, so a scene token must never be truncated.

The training token table (tools/release/setup/build_trainval_simscale_table.py) lists every training token, real or
synthetic, with the columns of TABLE_COLUMNS; SimScaleSceneLoader serves its synthetic rows.  Scenes are read lazily,
one pickle per call: 237k pickles are about 36 GB, too much to hold in every rank.  The pickles are downloaded data:
SceneUnpickler resolves only the numpy names a frame list is made of (as the table builder does).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from navsim.common.dataclasses import AgentInput, Scene, SensorConfig

ROUNDS = (0, 1, 2, 3, 4)
NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 8
NUM_FRAMES = NUM_HISTORY_FRAMES + NUM_FUTURE_FRAMES
CURRENT = NUM_HISTORY_FRAMES - 1
META_DIR = "meta_datas"
HIST_DIR = "sensor_blobs_hist"

# token table: one row per training token, row = position (the row order of every row-indexed asset)
TABLE_COLUMNS = ("token", "log", "city", "row", "source", "round", "pkl")
SOURCE_REAL = ("navtrain_train", "navtrain_val")
SOURCE_SIM = "simscale"


def round_dir(r: int) -> str:
    return f"synthetic_reaction_pdm_v1.0-{int(r)}"


def scene_pkl(r: int, log_stem: str) -> str:
    """pkl column: path of a scene pickle relative to SSR_SIMSCALE_ROOT."""
    return f"{round_dir(r)}/{META_DIR}/{log_stem}.pkl"


def sensor_root(root: Union[str, Path], r: int) -> Path:
    return Path(root) / round_dir(r) / HIST_DIR


class SceneUnpickler(pickle.Unpickler):
    """A SimScale scene pickle is downloaded data: only the numpy names an OpenScene frame list is made of resolve
    (the same set as tools/release/setup/build_trainval_simscale_table.SceneUnpickler)."""

    ALLOWED = {(m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")} | \
        {("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a scene pickle")
        return super().find_class(module, name)


def read_scene(root: Union[str, Path], pkl: str, token: str = None) -> List[Dict[str, Any]]:
    """The NUM_FRAMES frame dicts of one scene pickle; token, when given, must be its frames[CURRENT] token."""
    with open(Path(root) / pkl, "rb") as fh:
        frames = SceneUnpickler(fh).load()
    if len(frames) < NUM_FRAMES:
        raise ValueError(f"{pkl}: {len(frames)} frames, a SimScale scene has {NUM_FRAMES}")
    frames = frames[:NUM_FRAMES]
    if token is not None and frames[CURRENT]["token"] != token:
        raise ValueError(f"{pkl}: frame {CURRENT} token {frames[CURRENT]['token']} != {token}")
    return frames


class SimScaleSceneLoader:
    """SceneLoader interface over the synthetic rows of a token table, reading one scene pickle per call.

    rows: iterable of (token, pkl, round) or a DataFrame with those columns (other rows are ignored).  Picklable:
    it holds only the root, the row map and the sensor config, so DataLoader workers can fork it.
    """

    def __init__(self, root: Union[str, Path], rows, sensor_config: SensorConfig = SensorConfig.build_no_sensors()):
        self._root = Path(root)
        self._sensor_config = sensor_config
        if hasattr(rows, "itertuples"):
            rows = rows[rows["source"] == SOURCE_SIM] if "source" in rows else rows
            rows = zip(rows["token"].astype(str), rows["pkl"].astype(str), rows["round"].astype(int))
        self._rows: Dict[str, Tuple[str, int]] = {}
        for token, pkl, r in rows:
            if token in self._rows:
                raise ValueError(f"duplicate SimScale token {token}")
            self._rows[str(token)] = (str(pkl), int(r))
        self._tokens = list(self._rows)

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx) -> str:
        return self._tokens[idx]

    def __contains__(self, token: str) -> bool:
        return token in self._rows

    def scene_dict_list(self, token: str) -> List[Dict[str, Any]]:
        pkl, _ = self._rows[token]
        return read_scene(self._root, pkl, token)

    def sensor_blobs_path(self, token: str) -> Path:
        return sensor_root(self._root, self._rows[token][1])

    def get_scene_from_token(self, token: str) -> Scene:
        return Scene.from_scene_dict_list(
            self.scene_dict_list(token),
            self.sensor_blobs_path(token),
            num_history_frames=NUM_HISTORY_FRAMES,
            num_future_frames=NUM_FUTURE_FRAMES,
            sensor_config=self._sensor_config,
        )

    def get_agent_input_from_token(self, token: str, use_fut_frames: bool = False) -> AgentInput:
        if use_fut_frames:   # AgentInput.from_scene_dict_list assumes 10 future frames with images
            raise ValueError("SimScale scenes have no future images (use_fut_frames)")
        return AgentInput.from_scene_dict_list(
            self.scene_dict_list(token),
            self.sensor_blobs_path(token),
            num_history_frames=NUM_HISTORY_FRAMES,
            sensor_config=self._sensor_config,
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """{log: [token]}; a SimScale log is its scene pickle stem, one token each."""
        return {Path(pkl).stem: [token] for token, (pkl, _) in self._rows.items()}


def camera_files(frames: Sequence[Dict[str, Any]], cams: Iterable[str], frame_ids: Iterable[int]) -> List[str]:
    """data_path of the given cameras at the given frames (relative to the round's sensor root)."""
    return [frames[i]["cams"][c]["data_path"] for i in frame_ids for c in cams]
