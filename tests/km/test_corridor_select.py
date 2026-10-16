"""km 10/17: in-forward corridor selection (navsim/agents/para_ssr/modules/corridor_select.py)."""
import pytest
import torch

from navsim.agents.para_ssr.modules.corridor_select import corridor_select, parse_rule


def _preds(b=2, k=8, planes=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.zeros(k, 8, 3)
    anchors[:, :, 0] = torch.linspace(1, 8, 8)[None] * torch.linspace(0.5, 2.0, k)[:, None]   # straight, k speeds
    return {
        "trajectory_anchors": anchors,
        "trajectory_offset": torch.zeros(b, k, 8, 3),
        "plan_final_rewards": torch.randn(b, k, generator=g),
        "im_rewards": torch.softmax(torch.randn(b, k, generator=g), -1),
        "sim_rewards": torch.rand(b, 5, k, generator=g),
        "corridor_sdf": torch.full((b, 100, 200), 5.0),
        "corridor_occ": torch.full((b, planes, 100, 200), -20.0),          # logits: free everywhere
    }


def test_parse_rule():
    assert parse_rule("v2") == ("v2", None)
    assert parse_rule("pred_multiply_t0.25") == ("multiply", 0.25)
    assert parse_rule("pred_replace_t0.5") == ("replace", 0.5)
    for bad in ("v3_veto_ttchead", "pred_multiply", "multiply_t0.25"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_v2_rule_is_argmax():
    p = _preds()
    out = corridor_select(p, torch.zeros(2, 4), "v2")
    assert torch.equal(out["index"], p["plan_final_rewards"].argmax(1))


def test_free_corridor_keeps_v2_choice():
    # every candidate inside a free, drivable corridor -> the same corridor term for all -> v2's argmax
    p = _preds()
    out = corridor_select(p, torch.tensor([[2.0, 0, 0, 0]] * 2), "pred_multiply_t0.25")
    assert torch.equal(out["index"], p["plan_final_rewards"].argmax(1))
    assert torch.allclose(out["trajectory"], p["trajectory_anchors"][out["index"]])


def test_blocked_far_cells_move_choice_to_short_candidate():
    # occupancy on every cell beyond 12 m forward (footprint front = 4.05 m ahead of the rear axle): candidates
    # that reach it lose p_nc -> a shorter one wins
    p = _preds(b=1)
    p["plan_final_rewards"] = torch.linspace(0, 1, 8)[None]               # v2 prefers the longest
    p["corridor_occ"][:, :, int(12 / 0.32):] = 20.0
    out = corridor_select(p, torch.tensor([[1.0, 0, 0, 0]]), "pred_multiply_t0.25")
    assert out["index"].item() < 7
