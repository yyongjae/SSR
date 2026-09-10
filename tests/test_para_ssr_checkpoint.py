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


@pytest.mark.parametrize("checkpoint", [{}, {"state_dict": []}])
def test_initialize_rejects_invalid_checkpoint_schema(
    tmp_path: Path, checkpoint
) -> None:
    checkpoint_path = tmp_path / "invalid.ckpt"
    torch.save(checkpoint, checkpoint_path)

    agent = _minimal_agent(checkpoint_path)
    with pytest.raises(RuntimeError, match="checkpoint"):
        agent.initialize()
