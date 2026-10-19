from __future__ import annotations

import lzma
import pickle

from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from collections.abc import Mapping
from typing import Iterator

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache
import numpy as np

def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [
            log_file
            for log_file in log_files
            if log_file.name.replace(".pkl", "") in scene_filter.log_names
        ]

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(
            scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval
        ):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if (
                scene_filter.has_route
                and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0
            ):
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list

            if (scene_filter.max_scenes is not None) and (
                len(filtered_scenes) >= scene_filter.max_scenes
            ):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes



class SceneFrameStore(Mapping):
    """token -> list of frame dicts, held as ONE pickle byte buffer.

    Why not a plain dict: DataLoader workers are forked from the main process
    and share its memory copy-on-write.  A dict of ~85k scenes is millions of
    Python objects; merely reading one (even `token in tokens`) writes its
    refcount, so the kernel copies that page into the worker.  Over an epoch
    every worker grows its own copy of the metadata it touched (~1 GB/epoch
    measured), on top of 4 ranks x 8 workers (x2 while the val loader's
    workers overlap) -- enough to exhaust a 188 GB host.  One numpy array has
    a single refcount in a single page, so the buffer stays shared for the
    life of the worker; a scene is unpickled only when requested and freed
    right after.

    Frames are stored once each: with frame_interval=1 a frame belongs to up
    to num_frames overlapping scenes, so pickling per scene would duplicate
    the data ~14x (measured 16 GB for a 52k-scene subset).  A scene is a
    contiguous run of frames of one log, so it is just (start, count) into
    the frame table.
    """

    def __init__(self, scene_frames_dicts: Dict[str, List[Dict[str, Any]]]):
        self._tokens: List[str] = list(scene_frames_dicts.keys())
        self._index: Dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
        self._log_names: List[str] = [
            frames[0]["log_name"] for frames in scene_frames_dicts.values()
        ]
        frame_slot: Dict[int, int] = {}          # id(frame dict) -> frame index
        chunks: List[bytes] = []
        offsets: List[int] = [0]
        scene_start = np.zeros(len(self._tokens), dtype=np.int64)
        scene_count = np.zeros(len(self._tokens), dtype=np.int32)
        keepalive: List[Dict[str, Any]] = []   # ids are only unique while objects live
        for i, frames in enumerate(scene_frames_dicts.values()):
            slots = []
            for frame in frames:
                slot = frame_slot.get(id(frame))
                if slot is None:
                    slot = len(chunks)
                    chunk = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
                    chunks.append(chunk)
                    offsets.append(offsets[-1] + len(chunk))
                    frame_slot[id(frame)] = slot
                    keepalive.append(frame)
                slots.append(slot)
            assert slots == list(range(slots[0], slots[0] + len(slots))), (
                "scene frames are expected to be consecutive frames of one log"
            )
            scene_start[i], scene_count[i] = slots[0], len(slots)
        self._scene_start = scene_start
        self._scene_count = scene_count
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._buffer = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        self.num_frames = len(chunks)
        del chunks, keepalive, frame_slot

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @property
    def nbytes(self) -> int:
        return int(self._buffer.nbytes)

    def log_name_of(self, token: str) -> str:
        return self._log_names[self._index[token]]

    def _frame(self, slot: int) -> Dict[str, Any]:
        start, end = int(self._offsets[slot]), int(self._offsets[slot + 1])
        # memoryview slice: zero-copy view into the shared buffer
        return pickle.loads(self._buffer.data[start:end])

    def __getitem__(self, token: str) -> List[Dict[str, Any]]:
        i = self._index[token]
        start, count = int(self._scene_start[i]), int(self._scene_count[i])
        return [self._frame(s) for s in range(start, start + count)]

    def __contains__(self, token: object) -> bool:
        return token in self._index

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)


class SceneLoader:

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        train_debug: bool = False,
    ):
        if train_debug:
            print('begin loading scene_frames_dicts')
            scene_frames_dicts = np.load('/home/yingyan.li/repo/WoTE/dataset/scene_frames_dicts_debug.npy', allow_pickle=True).item()
            print('finish loading scene_frames_dicts')
        else:
            print('Begin filtering scenes')
            scene_frames_dicts = filter_scenes(data_path, scene_filter)
            print('Finish filtering Scenes ')
        # Pack into one shared byte buffer (see SceneFrameStore) and drop the
        # object graph so neither the main process nor forked workers hold it.
        self.scene_frames_dicts: SceneFrameStore = SceneFrameStore(scene_frames_dicts)
        del scene_frames_dicts
        print(f"Packed {len(self.scene_frames_dicts)} scenes / "
              f"{self.scene_frames_dicts.num_frames} frames into "
              f"{self.scene_frames_dicts.nbytes / 2**20:.0f} MB shared buffer")
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

    @property
    def tokens(self) -> List[str]:
        # Cached list: do not rebuild (and touch every key) on each access.
        return self.scene_frames_dicts.tokens

    def __len__(self):
        return len(self.scene_frames_dicts)

    def __getitem__(self, idx) -> str:
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        assert token in self.scene_frames_dicts
        return Scene.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
        )

    def get_agent_input_from_token(self, token: str, use_fut_frames=False) -> AgentInput:
        assert token in self.scene_frames_dicts
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
            use_fut_frames=use_fut_frames,
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token in self.tokens:
            # log_name is kept out of the pickle buffer so this needs no unpickling
            tokens_per_logs.setdefault(self.scene_frames_dicts.log_name_of(token), []).append(token)
        return tokens_per_logs

class MetricCacheLoader:

    def __init__(
        self,
        cache_path: Path,
        file_name: str = "metric_cache.pkl",
    ):

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths=f.read().splitlines()[1:]
        metric_cache_dict = {
            cache_path.split("/")[-2]: cache_path
            for cache_path in cache_paths
        }
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:

        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        return metric_cache

    def to_pickle(self, path: Path) -> None:
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
