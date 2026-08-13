"""Occupancy ground-truth rasterisation for PARA-SSR.

PARA-Drive attaches a scene-level occupancy module directly to the shared BEV
feature.  UniAD produces its occupancy GT by re-loading the annotations of the
future key-frames and chaining ego poses (``reframe_boxes``).  SSR does not need
any of that: the VAD-style ``.pkl`` already stores, for every agent of the
*current* frame,

    ``gt_agent_fut_trajs``  per-step (x, y) offsets, 6 steps @ 0.5 s
    ``gt_agent_fut_yaw``    per-step yaw deltas
    ``gt_agent_fut_masks``  per-step validity

all expressed in the **current frame's LiDAR coordinates** (see
``tools/data_converter/vad_nuscenes_converter.py:352-395``).  So the future
occupancy can be rasterised on the fly by displacing/rotating the current boxes.
No extra preprocessing, no extra files.

The raster grid is deliberately identical to SSR's BEV feature grid so that the
occupancy head is a plain convolutional decoder with no resampling:

    ``bev_embed[b, h * bev_w + w]``  <->  x = pc_range[0] + (w + .5) * dx
                                          y = pc_range[1] + (h + .5) * dy

i.e. rows index y (longitudinal), columns index x (lateral).  See
``BEVFormerEncoder.get_reference_points`` in ``SSR/modules/encoder.py``.
"""
import cv2
import numpy as np

from mmdet.datasets.builder import PIPELINES

# Indices into the 10 nuScenes detection classes used by VAD/SSR:
#   0 car, 1 truck, 2 construction_vehicle, 3 bus, 4 trailer,
#   5 barrier, 6 motorcycle, 7 bicycle, 8 pedestrian, 9 traffic_cone
# The "vehicle" group mirrors UniAD's ``only_vehicle`` 7-class set.
DEFAULT_OCC_CLASS_LABELS = (
    (0, 1, 2, 3, 4, 6, 7),  # vehicle
    (8, ),                  # pedestrian
)


@PIPELINES.register_module()
class GenerateSSROccLabels(object):
    """Rasterise current + future agent occupancy onto SSR's BEV grid.

    Args:
        pc_range (list[float]): ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
            Must match the model's ``point_cloud_range``.
        bev_size (tuple[int]): ``(bev_h, bev_w)``; rows are y, columns are x.
        n_future (int): number of future steps (0.5 s each).  The produced
            tensor has ``n_future + 1`` frames, frame 0 being the present.
        class_labels (tuple[tuple[int]]): one tuple of ``gt_labels_3d`` values
            per output channel.
        fut_ts (int): number of future steps stored in ``attr_labels`` (6).

    Produces:
        ``gt_occ_seg``   uint8 ``[n_future + 1, num_classes, bev_h, bev_w]``
        ``gt_occ_valid`` bool  ``[n_future + 1]`` -- frames whose GT is
            trustworthy.  Frames past the end of a scene are marked invalid so
            the loss can ignore them instead of learning "everything is empty".
    """

    def __init__(self,
                 pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
                 bev_size=(100, 100),
                 n_future=6,
                 class_labels=DEFAULT_OCC_CLASS_LABELS,
                 fut_ts=6):
        assert n_future <= fut_ts, \
            f'n_future ({n_future}) cannot exceed the {fut_ts} future steps ' \
            'stored in attr_labels'
        self.pc_range = list(pc_range)
        self.bev_h, self.bev_w = bev_size
        self.n_future = n_future
        self.num_frames = n_future + 1
        self.class_labels = [set(c) for c in class_labels]
        self.num_classes = len(self.class_labels)
        self.fut_ts = fut_ts

        # metres per cell; anisotropic for SSR (0.3 m lateral, 0.6 m longitudinal)
        self.dx = (self.pc_range[3] - self.pc_range[0]) / self.bev_w
        self.dy = (self.pc_range[4] - self.pc_range[1]) / self.bev_h

    def _poly_to_grid(self, cx, cy, length, width, yaw):
        """Rotated BEV footprint -> integer (col, row) polygon for cv2."""
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        # corners in the box frame: length along x, width along y
        corners = np.array([
            [length / 2, -length / 2, -length / 2, length / 2],
            [width / 2, width / 2, -width / 2, -width / 2],
        ])
        corners = rot @ corners + np.array([[cx], [cy]])  # (2, 4) in LiDAR xy

        cols = (corners[0] - self.pc_range[0]) / self.dx
        rows = (corners[1] - self.pc_range[1]) / self.dy
        # cv2 expects (x, y) == (col, row)
        return np.round(np.stack([cols, rows], axis=1)).astype(np.int32)

    def __call__(self, results):
        seg = np.zeros(
            (self.num_frames, self.num_classes, self.bev_h, self.bev_w),
            dtype=np.uint8)

        boxes = results['gt_bboxes_3d'].tensor.numpy()
        labels = np.asarray(results['gt_labels_3d'])
        # Placed before CustomObjectRangeFilter the key is still `attr_labels`;
        # after it, the filtered copy is `gt_attr_labels`.
        attr_key = 'gt_attr_labels' if 'gt_attr_labels' in results else 'attr_labels'
        attr = np.asarray(results[attr_key], dtype=np.float32)

        results['gt_occ_seg'] = seg
        results['gt_occ_valid'] = self._frame_validity(results)
        if boxes.shape[0] == 0:
            return results
        assert attr.shape[0] == boxes.shape[0], (
            f'{attr_key} ({attr.shape[0]}) and gt_bboxes_3d ({boxes.shape[0]}) '
            'are misaligned')

        T = self.fut_ts
        fut_trajs = np.cumsum(attr[:, :T * 2].reshape(-1, T, 2), axis=1)
        fut_masks = attr[:, T * 2:T * 3].reshape(-1, T)
        fut_yaw = np.cumsum(attr[:, T * 3 + 10:T * 4 + 10].reshape(-1, T), axis=1)

        # nuScenes stores size as (width, length, height); the VAD converter
        # writes it straight into columns 3:6 and encodes yaw in SECOND format.
        # Both conversions below mirror PlanningMetric.get_birds_eye_view_label
        # (SSR/planner/metric_stp3.py) so occupancy GT and the collision metric
        # rasterise agents identically.
        width, length = boxes[:, 3], boxes[:, 4]
        yaw0 = -1.0 * (boxes[:, 6] + np.pi / 2)
        xy0 = boxes[:, 0:2]

        # generous margin so agents straddling the border are still drawn
        x_lo, x_hi = self.pc_range[0] - 20.0, self.pc_range[3] + 20.0
        y_lo, y_hi = self.pc_range[1] - 20.0, self.pc_range[4] + 20.0

        channel_of = {}
        for ch, label_set in enumerate(self.class_labels):
            for lab in label_set:
                channel_of[lab] = ch

        for i in range(boxes.shape[0]):
            ch = channel_of.get(int(labels[i]), None)
            if ch is None:  # not an occupancy class (barrier, cone, ignored)
                continue
            for t in range(self.num_frames):
                if t == 0:
                    cx, cy, yaw = xy0[i, 0], xy0[i, 1], yaw0[i]
                else:
                    if fut_masks[i, t - 1] != 1:
                        break  # annotation chain ended; stop this agent
                    cx = xy0[i, 0] + fut_trajs[i, t - 1, 0]
                    cy = xy0[i, 1] + fut_trajs[i, t - 1, 1]
                    yaw = yaw0[i] + fut_yaw[i, t - 1]
                if not (x_lo < cx < x_hi and y_lo < cy < y_hi):
                    continue
                poly = self._poly_to_grid(cx, cy, length[i], width[i], yaw)
                cv2.fillPoly(seg[t, ch], [poly], 1)

        return results

    def _frame_validity(self, results):
        """Future frames beyond the end of a scene carry no usable GT."""
        valid = np.ones(self.num_frames, dtype=bool)
        ego_fut_masks = results.get('ego_fut_masks', None)
        if ego_fut_masks is not None:
            m = np.asarray(ego_fut_masks).reshape(-1).astype(bool)
            n = min(self.n_future, m.shape[0])
            valid[1:1 + n] = m[:n]
            if n < self.n_future:
                valid[1 + n:] = False
        elif not results.get('fut_valid_flag', True):
            valid[1:] = False
        return valid

    def __repr__(self):
        return (f'{self.__class__.__name__}(pc_range={self.pc_range}, '
                f'bev_size=({self.bev_h}, {self.bev_w}), '
                f'n_future={self.n_future}, num_classes={self.num_classes})')
