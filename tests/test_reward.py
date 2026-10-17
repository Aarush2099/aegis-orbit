import numpy as np

from benchmarks.oracle_reference import load_config
from benchmarks.run_unshielded_benchmark import step_reward_and_termination


def test_collision_penalty_is_one_time_and_collision_terminates():
    config = load_config()
    info = {"min_distance_in_step": 9.9, "distance": 9.9}
    action = np.zeros(3)

    first, done, collided = step_reward_and_termination(
        action,
        info,
        config,
        prior_collided=False,
        at_horizon=False,
        proximity_weight=0.0,
        clipped_action=action,
    )
    repeated, repeated_done, repeated_collided = step_reward_and_termination(
        action,
        info,
        config,
        prior_collided=collided,
        at_horizon=False,
        proximity_weight=0.0,
        clipped_action=action,
    )

    assert first["collision_term"] == -100.0
    assert repeated["collision_term"] == 0.0
    assert done and repeated_done
    assert collided and repeated_collided


def test_survival_bonus_only_applies_to_non_collision_horizon():
    config = load_config()
    action = np.zeros(3)
    safe_info = {"min_distance_in_step": 20.0, "distance": 20.0}
    collision_info = {"min_distance_in_step": 9.9, "distance": 9.9}

    survived, done, collided = step_reward_and_termination(
        action,
        safe_info,
        config,
        prior_collided=False,
        at_horizon=True,
        proximity_weight=0.0,
        clipped_action=action,
    )
    collided_at_horizon, collision_done, collision_state = step_reward_and_termination(
        action,
        collision_info,
        config,
        prior_collided=False,
        at_horizon=True,
        proximity_weight=0.0,
        clipped_action=action,
    )

    assert done and not collided
    assert survived["survival_term"] == 10.0
    assert collision_done and collision_state
    assert collided_at_horizon["survival_term"] == 0.0
