import numpy as np

from aegis_orbit.agents import SACAgent
from benchmarks.oracle_reference import get, load_config
from benchmarks.run_unshielded_benchmark import deterministic_evaluation_policy


def test_evaluation_policy_returns_same_action_for_same_state():
    config = load_config()
    agent = SACAgent(
        state_dim=12,
        action_dim=3,
        u_max=float(get(config, "environment", "u_max")),
        learning_rate=float(get(config, "training", "actor_lr")),
        gamma=float(get(config, "training", "gamma")),
        tau=float(get(config, "training", "tau")),
        target_entropy=float(get(config, "training", "target_entropy")),
    )
    policy = deterministic_evaluation_policy(agent)
    state = np.random.default_rng(0).normal(size=12).astype(np.float32)

    first_action = policy(state, np.random.default_rng(1))
    second_action = policy(state, np.random.default_rng(2))

    assert np.array_equal(first_action, second_action)
