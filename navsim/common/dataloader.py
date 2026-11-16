from __future__ import annotations

import lzma
import pickle

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache
import numpy as np

def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    # filter logs
    log_files = sorted(data_path.iterdir())     # iterdir order is the filesystem's: scene order must not depend on it
    if scene_filter.log_names is not None:
        log_names = set(scene_filter.log_names)     # a set: list membership is O(N) per log
        log_files = [
            log_file
            for log_file in log_files
            if log_file.name.replace(".pkl", "") in log_names
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
            self.scene_frames_dicts = np.load('/home/yingyan.li/repo/WoTE/dataset/scene_frames_dicts_debug.npy', allow_pickle=True).item()
            print('finish loading scene_frames_dicts')
        else:
            print('Begin filtering scenes')
            self.scene_frames_dicts = filter_scenes(data_path, scene_filter)
            print('Finish filtering Scenes ')
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

    @property
    def tokens(self) -> List[str]:
        return list(self.scene_frames_dicts.keys())

    def __len__(self):
        return len(self.tokens)

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
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})
        return tokens_per_logs


class MultiSceneLoader:
    """SceneLoader interface over several scene loaders, in one fixed token order (run_training's table-driven training
    set: a SceneLoader of the real rows and the navsim.common.simscale.SimScaleSceneLoader of the synthetic rows of one
    token table, tokens in table row order).

    Every token of `tokens` must be served by exactly one loader: a token that its loader's filter dropped (a SimScale
    scene under the 4 + 10 navtrain window, a real token without a route, ...) raises here instead of shrinking the
    training set.  Scenes come from the owning loader per call, so a lazy loader stays lazy.  tokens is one cached
    list (Dataset.__getitem__ indexes it per sample).  restrict_tokens is the teacher-cache filter's way in (this class
    has no scene_frames_dicts: the synthetic scenes are not in memory).  Picklable when its loaders are."""

    def __init__(self, tokens: Sequence[str], loaders: Sequence[Any]):
        self._loaders = list(loaders)
        owner: Dict[str, int] = {}
        for i, loader in enumerate(self._loaders):
            for token in loader.tokens:
                if token in owner:
                    raise ValueError(f"token {token} is served by loader {owner[token]} and loader {i}")
                owner[token] = i
        self._tokens = [str(t) for t in tokens]
        if len(set(self._tokens)) != len(self._tokens):
            raise ValueError(f"{len(self._tokens) - len(set(self._tokens))} duplicate tokens")
        missing = [t for t in self._tokens if t not in owner]
        if missing:
            raise ValueError(f"{len(missing)} of {len(self._tokens)} tokens are served by no loader (dropped by a "
                             f"scene filter?), e.g. {missing[:3]}")
        self._owner = {t: owner[t] for t in self._tokens}

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx) -> str:
        return self._tokens[idx]

    def __contains__(self, token: str) -> bool:
        return token in self._owner

    def loader_of(self, token: str) -> Any:
        return self._loaders[self._owner[token]]

    def get_scene_from_token(self, token: str) -> Scene:
        return self.loader_of(token).get_scene_from_token(token)

    def get_agent_input_from_token(self, token: str, use_fut_frames=False) -> AgentInput:
        return self.loader_of(token).get_agent_input_from_token(token, use_fut_frames=use_fut_frames)

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        tokens_per_logs: Dict[str, List[str]] = {}
        for loader in self._loaders:
            for log_name, tokens in loader.get_tokens_list_per_log().items():
                tokens = [t for t in tokens if t in self._owner]
                if tokens:
                    tokens_per_logs.setdefault(log_name, []).extend(tokens)
        return tokens_per_logs

    def restrict_tokens(self, keep: Callable[[str], bool]) -> int:
        """Keep the tokens with keep(token) true, in order -> how many were dropped."""
        before = len(self._tokens)
        self._tokens = [t for t in self._tokens if keep(t)]
        self._owner = {t: self._owner[t] for t in self._tokens}
        return before - len(self._tokens)


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
