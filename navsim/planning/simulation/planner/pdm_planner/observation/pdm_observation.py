from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Polygon

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_object_manager import (
    PDMObjectManager,
)
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        map_radius: float,
        observation_sample_res: int = 2,
        extend_observation_for_ttc: bool = True,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        :param extend_observation_for_ttc: observe the proposal horizon + 1 s (TTC look-ahead), defaults to True.
            False (NAVSIM v2 name): the longer of the two horizons only, for a GT future that ends there (SimScale
            metric cache, 4 s); extend_constant_velocity then adds the TTC look-ahead past it.
        """
        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "PDMObservation: Proposals and Trajectory must have equal interval length!"

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_sampling.interval_length  # [s]

        if extend_observation_for_ttc:
            self._observation_samples: int = (
                proposal_sampling.num_poses + int(1 / self._sample_interval)
                if proposal_sampling.num_poses + int(1 / self._sample_interval)
                > trajectory_sampling.num_poses
                else trajectory_sampling.num_poses
            )
        else:
            self._observation_samples: int = max(trajectory_sampling.num_poses, proposal_sampling.num_poses)

        self._map_radius: float = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res
            for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: Optional[List[PDMOccupancyMap]] = None
        self._unique_objects: Optional[Dict[str, TrackedObject]] = None

        self._initialized: bool = False

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert (
            0 <= time_idx < len(self._global_to_local_idcs)
        ), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    def __len__(self) -> int:
        """
        Number of 10Hz time indices __getitem__ serves (0 .. len - 1): 51 for a metric cache (a SimScale one after
        extend_constant_velocity too), 41 for a 4 s one before it.
        Computed from attributes that every pickled observation has, so older metric caches answer it too.
        :return: number of time indices
        """
        return len(self._global_to_local_idcs)

    @property
    def extrapolated_from(self) -> Optional[int]:
        """
        First time index that extend_constant_velocity forecast rather than the GT future gave, None for an observation
        it did not extend (every 5 s metric cache, and caches pickled before the method existed).
        :return: time index or None
        """
        return getattr(self, "_extrapolated_from", None)

    def extend_constant_velocity(self, observation_samples: int) -> int:
        """
        Extends a GT observation that ends before observation_samples, i.e. a GT future shorter than the proposal
        horizon + 1 s of TTC look-ahead (SimScale metric cache, 4 s): every object of the last occupancy map moves on at
        its velocity over the last interval (its polygon centroid since the map before; the GT is interpolated linearly,
        so the velocity of the last GT interval), and an object not in the map before keeps its place.  PDMScorer then
        scores the TTC look-ahead with the same rule as for a 5 s cache.  Only for observation_sample_res 1.
        :param observation_samples: number of 10Hz samples after the current one to observe
        :return: number of occupancy maps added, 0 when the observation reaches observation_samples already
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert self._observation_sample_res == 1, "PDMObservation: extension needs observation_sample_res 1"
        n_add = observation_samples - self._observation_samples
        if n_add <= 0:
            return 0
        last, before = self._occupancy_maps[-1], self._occupancy_maps[-2]
        tokens = list(last.tokens)
        geometries = np.array([last[token] for token in tokens], dtype=np.object_)
        dxy = np.zeros((len(tokens), 2), dtype=np.float64)
        for idx, token in enumerate(tokens):
            if token in before.token_to_idx:
                now, then = geometries[idx].centroid, before[token].centroid
                dxy[idx] = (now.x - then.x, now.y - then.y)
        _, owner = shapely.get_coordinates(geometries, return_index=True)
        step_xy = dxy[owner]                                    # per coordinate, in get_coordinates order
        for step in range(1, n_add + 1):
            moved = shapely.transform(geometries, lambda xy, offset=step_xy * step: xy + offset)
            self._occupancy_maps.append(PDMOccupancyMap(tokens, moved))
        self._extrapolated_from = self._observation_samples + 1
        self._observation_samples = observation_samples
        self._global_to_local_idcs = list(range(observation_samples + 1))
        return n_add

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: string
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._unique_objects

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        object_manager = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(dynamic_object_coords_t)

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(self._occupancy_maps[0][intersecting_obstacle])
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._unique_objects = object_manager.unique_objects
        self._initialized = True

    def update_replay(self, scenario: AbstractScenario, iteration_index: int) -> None:
        detection_tracks = scenario.get_future_tracked_objects(
            iteration_index, self._observation_samples * self._sample_interval
        )
        occupancy_maps = []
        unique_objects = {}

        for detection_track in detection_tracks:
            tokens, polygons = [], []
            for tracked_object in detection_track.tracked_objects:
                token, polygon = tracked_object.track_token, tracked_object.box.geometry
                tokens.append(token)
                polygons.append(polygon)

                if token not in unique_objects.keys():
                    unique_objects[token] = tracked_object

            occupancy_map = PDMOccupancyMap(tokens, polygons)
            occupancy_maps.append(occupancy_map)

        assert (
            len(occupancy_maps) == self._observation_samples + 1
        ), f"Expected observation length {self._observation_samples + 1}, but got {len(occupancy_maps)}"

        self._occupancy_maps: List[PDMOccupancyMap] = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def update_detections_tracks(self, detection_tracks: List[DetectionsTracks]) -> None:
        occupancy_maps = []
        unique_objects = {}

        for detection_track in detection_tracks:
            tokens, polygons = [], []
            for tracked_object in detection_track.tracked_objects:
                token, polygon = tracked_object.track_token, tracked_object.box.geometry
                tokens.append(token)
                polygons.append(polygon)

                if token not in unique_objects.keys():
                    unique_objects[token] = tracked_object

            occupancy_map = PDMOccupancyMap(tokens, polygons)
            occupancy_maps.append(occupancy_map)

        assert (
            len(occupancy_maps) == self._observation_samples + 1
        ), f"Expected observation length {self._observation_samples + 1}, but got {len(occupancy_maps)}"

        self._occupancy_maps: List[PDMOccupancyMap] = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def _get_object_manager(
        self, ego_state: EgoState, observation: Observation
    ) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state: state of ego-vehicle
        :param observation: input observation of nuPlan
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for object in observation.tracked_objects:
            if (
                (object.tracked_object_type == TrackedObjectType.EGO)
                or (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._collided_track_ids)
            ):
                continue

            object_manager.add_object(object)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_data:
            lane_connector_id = str(data.lane_connector_id)

            if (data.status == TrafficLightStatusType.RED) and (
                lane_connector_id in route_lane_dict.keys()
            ):
                lane_connector = route_lane_dict[lane_connector_id]
                traffic_light_tokens.append(f"{self._red_light_token}_{lane_connector_id}")
                traffic_light_polygons.append(lane_connector.polygon)

        return traffic_light_tokens, traffic_light_polygons
