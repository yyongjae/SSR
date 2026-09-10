from types import SimpleNamespace

import pytest
import torch

from navsim.agents.para_ssr.modules.grad_balance import GradBalancer
from navsim.agents.para_ssr.para_ssr_loss import ParaSSRLoss


def _loss_config(target=None, **overrides):
    values = {
        "grad_balance_target": target,
        "grad_balance_interval": 1,
        "grad_balance_momentum": 0.9,
        "grad_balance_clamp": (1e-5, 1.0),
        "grad_balance_warmup_iters": 0,
        "grad_norm_log_interval": 0,
        "task_loss_weight": {"plan": 1.0, "det": 1.0, "motion": 1.0},
        "heading_weight": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _NoHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.det_motion_head = None
        self.map_head = None
        self.aux_grad_scale = {"det": 1.0, "map": 1.0}


class _CancellingDetMotionHead:
    def loss(self, predictions, *args):
        bev = predictions["bev_embed"]
        return {
            "loss_bbox": bev.sum(),
            "loss_traj_reg": -bev.sum(),
        }


def _plan_io():
    predictions = {
        "ego_fut_preds": torch.zeros(1, 1, 1, 2, requires_grad=True),
        "bev_embed": torch.ones(1, 2, requires_grad=True),
    }
    targets = {
        "trajectory_offsets": torch.zeros(1, 1, 2),
        "trajectory_mask": torch.ones(1, 1),
        "command": torch.ones(1, 1),
    }
    return predictions, targets


def test_validation_does_not_advance_balancer_iteration():
    loss_fn = ParaSSRLoss(_loss_config())
    model = _NoHeadModel()
    predictions, targets = _plan_io()

    model.eval()
    loss_fn(model, {}, targets, predictions)
    assert loss_fn.iteration == 0

    model.train()
    loss_fn(model, {}, targets, predictions)
    assert loss_fn.iteration == 1


def test_balancer_state_round_trips_through_module_state_dict():
    cfg = _loss_config({"plan": 0.4, "det": 0.3, "map": 0.3})
    loss_fn = ParaSSRLoss(cfg)
    loss_fn.iteration = 123
    loss_fn.balancer.scale.update(det=0.25, map=0.5)
    loss_fn.balancer._seen.update({"det", "map"})

    restored = ParaSSRLoss(cfg)
    restored.load_state_dict(loss_fn.state_dict())

    assert restored.iteration == 123
    assert restored.balancer.scale == {"det": 0.25, "map": 0.5}
    assert restored.balancer._seen == {"det", "map"}


def test_zero_target_scale_is_applied_before_first_forward():
    loss_fn = ParaSSRLoss(
        _loss_config({"plan": 1.0, "det": 0.0, "map": 1.0})
    )
    model = _NoHeadModel()

    loss_fn.apply_aux_scales(model)

    assert model.aux_grad_scale == {"det": 0.0, "map": 1.0}


def test_detection_and_motion_are_measured_as_one_valve_gradient():
    loss_fn = ParaSSRLoss(
        _loss_config({"plan": 0.5, "det": 0.5}, grad_norm_log_interval=1)
    )
    loss_fn.iteration = 1
    model = _NoHeadModel()
    model.det_motion_head = _CancellingDetMotionHead()
    predictions, targets = _plan_io()
    predictions["all_cls_scores"] = torch.empty(0)
    targets.update(
        gt_boxes=torch.empty(1, 0, 9),
        gt_labels=torch.empty(1, 0, dtype=torch.long),
        gt_valid=torch.empty(1, 0, dtype=torch.bool),
    )

    _, logs = loss_fn(model, {}, targets, predictions)

    assert logs["gnorm/det"].item() == pytest.approx(0.0)
    assert "gnorm/motion" not in logs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": 0},
        {"momentum": 1.0},
        {"clamp": (0.0, 1.0)},
        {"warmup_iters": -1},
    ],
)
def test_invalid_balancer_schedule_fails_fast(kwargs):
    with pytest.raises(ValueError):
        GradBalancer({"plan": 1.0, "det": 1.0}, **kwargs)


@pytest.mark.parametrize("key", ["motion", "occ"])
def test_unsupported_valve_target_fails_fast(key):
    with pytest.raises(ValueError, match="unsupported"):
        ParaSSRLoss(_loss_config({"plan": 1.0, key: 1.0}))
