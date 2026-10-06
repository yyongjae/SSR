"""Planning-side map consistency (navsim/agents/para_ssr/plan_map.py)."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon, box

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.para_ssr_loss import ParaSSRLoss
from navsim.agents.para_ssr.plan_map import (
    EGO_FRONT, EGO_HALF_WIDTH, EGO_REAR, DrivableAreaTargetBuilder, footprint_corners,
    grid_shape, plan_map_loss, rasterize_sdf, sample_sdf,
)

EXTENT = (-8.0, 72.0, -32.0, 32.0)
RES = 0.25


def _sdf(poly, clip=10.0):
    return torch.from_numpy(rasterize_sdf(poly, EXTENT, RES, clip))


def _at(sdf, x, y):
    val, on = sample_sdf(sdf[None], torch.tensor([[[x, y]]], dtype=torch.float32), EXTENT)
    assert bool(on.all())
    return float(val)


def test_sdf_sign_distance_holes_and_clip():
    road = box(-8, -3, 72, 3)                                   # x forward, y left
    sdf = _sdf(road)
    assert tuple(sdf.shape) == grid_shape(EXTENT, RES) == (320, 256)
    assert _at(sdf, 10, 0) == pytest.approx(3.0, abs=0.2)       # centre: 3 m to either edge
    assert _at(sdf, 10, 2) == pytest.approx(1.0, abs=0.2)
    assert _at(sdf, 10, 5) == pytest.approx(-2.0, abs=0.2)      # 2 m off the left edge
    assert _at(sdf, 10, -20) == pytest.approx(-10.0, abs=1e-4)  # clipped
    holed = Polygon(box(0, -10, 40, 10).exterior.coords, [box(15, -2, 25, 2).exterior.coords])
    sdf = _sdf(holed)
    assert _at(sdf, 20, 0) < -1.5 and _at(sdf, 5, 0) > 4.5
    assert float(_sdf(None).max()) == pytest.approx(-10.0)      # no map: everything outside


def test_footprint_corners_follow_the_rear_axle_pose():
    c = footprint_corners(torch.tensor([[0.0, 0.0, 0.0]]))[0]
    expect = torch.tensor([[EGO_FRONT, EGO_HALF_WIDTH], [EGO_FRONT, -EGO_HALF_WIDTH],
                           [-EGO_REAR, EGO_HALF_WIDTH], [-EGO_REAR, -EGO_HALF_WIDTH]])
    assert torch.allclose(c, expect, atol=1e-6)
    c = footprint_corners(torch.tensor([[1.0, 2.0, np.pi / 2]]))[0]
    assert torch.allclose(c[0], torch.tensor([1.0 - EGO_HALF_WIDTH, 2.0 + EGO_FRONT]), atol=1e-5)


def _traj(y_end, T=8, speed=4.0):
    """Per-step offsets [T, 3] driving forward and drifting laterally to y_end."""
    xs = torch.arange(1, T + 1, dtype=torch.float32) * speed
    ys = torch.linspace(0, y_end, T + 1)[1:]
    poses = torch.stack([xs, ys, torch.zeros(T)], dim=1)
    return torch.diff(torch.cat([torch.zeros(1, 3), poses]), dim=0)


def _loss(pred, gt, sdf, cmd_index=1, branch=None, margin=0.0):
    preds = torch.zeros(1, 4, 8, 3)
    preds[0, :] = _traj(0.0)
    preds[0, cmd_index if branch is None else branch] = pred
    preds.requires_grad_(True)
    command = torch.eye(4)[cmd_index][None]
    loss, metrics = plan_map_loss(preds, command, gt[None], torch.ones(1, 8), sdf[None], EXTENT, margin)
    return loss, metrics, preds


def test_plan_map_loss_charges_leaving_the_road_and_pushes_back():
    sdf = _sdf(box(-8, -4, 72, 4))
    gt = _traj(0.0)
    loss, m, _ = _loss(_traj(0.0), gt, sdf)
    assert float(loss) == 0.0 and float(m["plan_map/pred_outside"]) == 0.0
    loss, m, preds = _loss(_traj(6.0), gt, sdf)
    assert float(loss) > 0 and float(m["plan_map/pred_outside"]) > 0
    loss.backward()
    assert float(preds.grad[0, 1, :, 1].sum()) > 0             # descent moves y back towards the road
    assert float(preds.grad[0, [0, 2, 3]].abs().sum()) == 0.0    # other command branches untouched


def test_plan_map_loss_never_argues_with_the_demonstration():
    sdf = _sdf(box(-8, -4, 72, 4))
    gt = _traj(6.0)                                             # the human left the road too
    loss, m, _ = _loss(_traj(6.0), gt, sdf)
    assert float(m["plan_map/gt_outside"]) > 0
    charged_only_inside = float(loss)
    loss_inside_gt, _, _ = _loss(_traj(6.0), _traj(0.0), sdf)
    assert charged_only_inside < float(loss_inside_gt)


def test_plan_map_loss_skips_corners_off_the_grid():
    sdf = _sdf(box(-8, -4, 72, 4))
    fast = _traj(8.0, speed=15.0)                               # beyond x1 after ~5 steps
    loss, m, _ = _loss(fast, fast, sdf)
    assert torch.isfinite(loss)


class _Obj(SimpleNamespace):
    pass


class _MapAPI:
    def __init__(self):
        self.asked = None

    def get_proximal_map_objects(self, point, radius, layers):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        self.asked = (radius, set(layers))
        lane = _Obj(polygon=box(-20, -2, 100, 2))
        return {
            L.ROADBLOCK: [_Obj(polygon=box(-20, -2, 100, 2), interior_edges=[lane])],
            L.ROADBLOCK_CONNECTOR: [_Obj(polygon=box(0, 0, 1, 1), interior_edges=[_Obj(polygon=box(30, 2, 40, 12))])],
            L.INTERSECTION: [],
            L.CARPARK_AREA: [_Obj(polygon=box(50, -30, 60, -20))],
        }


def _scene(map_api, heading=0.0):
    frame = SimpleNamespace(ego_status=SimpleNamespace(ego_pose=np.array([0.0, 0.0, heading])), token="t")
    return SimpleNamespace(map_api=map_api, frames=[frame, frame, frame, frame],
                           scene_metadata=SimpleNamespace(num_history_frames=4, map_name="m"))


def test_target_builder_uses_the_dac_layers_in_the_ego_frame():
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    cfg = replace(ParaSSRConfig(), plan_map_weight=1.0)
    api = _MapAPI()
    out = DrivableAreaTargetBuilder(cfg).compute_targets(_scene(api))
    sdf = out["drivable_sdf"].float()
    assert out["drivable_sdf"].dtype == torch.float16
    assert api.asked[1] == {L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA}
    assert api.asked[0] >= np.hypot(72, 32)
    assert _at(sdf, 10, 0) > 1.5                               # roadblock
    assert _at(sdf, 35, 7) > 1.5                               # lane connector of a ROADBLOCK_CONNECTOR
    assert _at(sdf, 0.5, 0.5 + 2.5) < 0                         # the connector's own polygon is NOT drivable
    assert _at(sdf, 55, -25) > 1.5                             # car park
    # rotated ego: the same road lies along the ego's y axis
    sdf = DrivableAreaTargetBuilder(cfg).compute_targets(_scene(api, heading=np.pi / 2))["drivable_sdf"].float()
    assert _at(sdf, 0, -10) > 1.5 and _at(sdf, 10, 0) < 0


class _NoHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.det_motion_head = None
        self.map_head = None
        self.aux_grad_scale = {}


def test_loss_adds_the_term_only_when_enabled_and_requires_the_target():
    base = replace(ParaSSRConfig(), use_det_motion_head=False, use_map_head=False,
                   grad_balance_target={"plan": 1.0}, grad_norm_log_interval=0)
    sdf = _sdf(box(-8, -4, 72, 4))
    preds = torch.zeros(1, 4, 8, 3)
    preds[0, 1] = _traj(6.0)
    preds.requires_grad_(True)
    targets = {"trajectory_offsets": _traj(0.0)[None], "trajectory_mask": torch.ones(1, 8),
               "command": torch.eye(4)[1][None], "drivable_sdf": sdf[None]}
    bev = torch.zeros(1, 10, 4, requires_grad=True)
    p = {"ego_fut_preds": preds, "bev_embed": bev}
    off, logs_off = ParaSSRLoss(base)(_NoHeadModel().train(), {}, targets, p)
    on, logs_on = ParaSSRLoss(replace(base, plan_map_weight=1.0))(_NoHeadModel().train(), {}, targets, p)
    assert "loss_plan_map" not in logs_off and float(logs_on["loss_plan_map"]) > 0
    assert float(on) == pytest.approx(float(off) + float(logs_on["loss_plan_map"]), rel=1e-5)
    with pytest.raises(KeyError):
        ParaSSRLoss(replace(base, plan_map_weight=1.0))(
            _NoHeadModel().train(), {}, {k: v for k, v in targets.items() if k != "drivable_sdf"}, p)
