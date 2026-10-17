import torch

from aegis_orbit.agents.sac_agent import SACAgent


def test_sac_update_moves_targets_and_keeps_temperature_positive():
    agent = SACAgent(12, 3, 1e-4)
    batch = (
        torch.randn(32, 12), torch.randn(32, 3) * 1e-4,
        torch.randn(32), torch.randn(32, 12), torch.zeros(32),
    )
    before = next(agent.target1.parameters()).detach().clone()
    metrics = agent.update(batch)
    after = next(agent.target1.parameters()).detach()
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(before, after)
    assert agent.alpha.item() > 0.0