from pathlib import Path

import pytest
import torch

from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent


def _minimal_agent(checkpoint_path: Path) -> ParaSSRAgent:
    """Build only enough of the agent to exercise its checkpoint loader."""
    agent = ParaSSRAgent.__new__(ParaSSRAgent)
    torch.nn.Module.__init__(agent)
    agent.probe = torch.nn.Linear(2, 1)
    agent._checkpoint_path = str(checkpoint_path)
    return agent


def test_initialize_loads_exact_lightning_agent_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "exact.ckpt"
    expected_weight = torch.tensor([[2.0, -3.0]])
    expected_bias = torch.tensor([0.5])
    torch.save(
        {
            "state_dict": {
                "agent.probe.weight": expected_weight,
                "agent.probe.bias": expected_bias,
            }
        },
        checkpoint_path,
    )

    agent = _minimal_agent(checkpoint_path)
    agent.initialize()

    torch.testing.assert_close(agent.probe.weight, expected_weight)
    torch.testing.assert_close(agent.probe.bias, expected_bias)


def test_initialize_rejects_incompatible_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "missing_head.ckpt"
    torch.save(
        {"state_dict": {"agent.probe.weight": torch.ones(1, 2)}},
        checkpoint_path,
    )

    agent = _minimal_agent(checkpoint_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        agent.initialize()


def _agent_with_anchor_planner(checkpoint_path: Path) -> ParaSSRAgent:
    """The v2 module the evaluation forward pass needs, under its real parameter prefix."""
    agent = _minimal_agent(checkpoint_path)
    agent.para_ssr_model = torch.nn.Module()
    agent.para_ssr_model.pts_bbox_head = torch.nn.Module()
    agent.para_ssr_model.pts_bbox_head.anchor_planner = torch.nn.Linear(2, 1)
    return agent


def test_initialize_in_evaluation_rejects_a_checkpoint_without_the_anchor_planner(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "v1_only.ckpt"
    torch.save(
        {"state_dict": {"agent.probe.weight": torch.ones(1, 2), "agent.probe.bias": torch.zeros(1)}},
        checkpoint_path,
    )
    # evaluation (bare initialize): the module the forward pass needs is missing -> fail, no random weights
    with pytest.raises(RuntimeError, match="anchor_planner"):
        _agent_with_anchor_planner(checkpoint_path).initialize()
    # training init may start the new module fresh, but only when asked to
    agent = _agent_with_anchor_planner(checkpoint_path)
    agent.initialize(allow_fresh_modules=True)
    torch.testing.assert_close(agent.probe.weight, torch.ones(1, 2))


@pytest.mark.parametrize("checkpoint", [{}, {"state_dict": []}])
def test_initialize_rejects_invalid_checkpoint_schema(
    tmp_path: Path, checkpoint
) -> None:
    checkpoint_path = tmp_path / "invalid.ckpt"
    torch.save(checkpoint, checkpoint_path)

    agent = _minimal_agent(checkpoint_path)
    with pytest.raises(RuntimeError, match="checkpoint"):
        agent.initialize()
