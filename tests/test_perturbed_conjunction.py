import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aegis_orbit.envs import ContinuousConjunctionEnv, PerturbedConjunctionEnv
from aegis_orbit.envs.cw_relative_dynamics import relative_drift_acceleration


def _kwargs():
    return dict(n=1.1e-3, dt=1.0, horizon_steps=10, u_max=1e-4, d_safe=100.0, initial_offset_m=200.0, closing_speed_mps=0.5)


def test_disabled_wrapper_matches_base_exactly():
    base = ContinuousConjunctionEnv(**_kwargs())
    wrapper = PerturbedConjunctionEnv(**_kwargs(), allow_unvalidated_d_safe=True)
    seed = 3
    base.reset(np.random.default_rng(seed))
    wrapper.reset(np.random.default_rng(seed))
    for control in (np.zeros(3), np.array([1e-5, -2e-5, 3e-5])):
        expected, _ = base.step(control)
        actual, info = wrapper.step(control)
        assert np.allclose(actual, expected, atol=1e-12)
        assert np.array_equal(info["perturbation_accel"], np.zeros(3))
        assert np.isfinite(info["min_distance_in_step"])


def test_cw_j2_truth_can_keep_cw_only_internal_shield_drift():
    env = PerturbedConjunctionEnv(
        **_kwargs(),
        j2_enabled=True,
        truth_model="cw_j2",
        shield_model="cw",
        allow_unvalidated_d_safe=True,
    )
    state = env.reset(np.random.default_rng(2))

    assert np.allclose(
        env._shield_drift(state),
        relative_drift_acceleration(state[:6], env.n),
        rtol=0.0,
        atol=0.0,
    )


def test_model_switches_are_reported_and_configured():
    env = PerturbedConjunctionEnv(**_kwargs(), j2_enabled=True, truth_model="cw_j2", shield_model="cw_j2", allow_unvalidated_d_safe=True)
    env.reset(np.random.default_rng(0))
    _, info = env.step(np.zeros(3))
    assert info["truth_model"] == "cw_j2"
    assert info["shield_model"] == "cw_j2"
    assert env.shield.drift_fn is not None


@pytest.mark.parametrize("truth_model", ["cw_j2_drag", "perturbed", "invalid"])
def test_truth_model_rejects_values_outside_enum(truth_model):
    with pytest.raises(ValueError, match="truth_model and shield_model must be cw or cw_j2"):
        PerturbedConjunctionEnv(
            **_kwargs(),
            truth_model=truth_model,
            allow_unvalidated_d_safe=True,
        )


def test_storm_none_matches_multiplier_one():
    common = _kwargs()
    first = PerturbedConjunctionEnv(**common, drag_enabled=True, ballistic_coeff_chief=2.0, ballistic_coeff_deputy=3.0, allow_unvalidated_d_safe=True)
    second = PerturbedConjunctionEnv(**common, drag_enabled=True, storm={"onset_s": 0.0, "rise_s": 1.0, "peak_factor": 1.0, "decay_s": 1.0}, ballistic_coeff_chief=2.0, ballistic_coeff_deputy=3.0, allow_unvalidated_d_safe=True)
    first.reset(np.random.default_rng(4))
    second.reset(np.random.default_rng(4))
    first_state, first_info = first.step(np.zeros(3))
    second_state, second_info = second.step(np.zeros(3))
    assert np.allclose(first_state, second_state, atol=1e-12)
    assert np.allclose(first_info["perturbation_accel"], second_info["perturbation_accel"], atol=1e-20)


def test_splitting_matches_full_rk45_one_step():
    env = PerturbedConjunctionEnv(
        **_kwargs(), j2_enabled=True, truth_model="cw_j2", drag_enabled=True,
        ballistic_coeff_chief=2.0, ballistic_coeff_deputy=3.0, allow_unvalidated_d_safe=True,
    )
    initial = env.reset(np.random.default_rng(5))
    control = np.array([2e-5, -1e-5, 3e-5], dtype=np.float64)

    def derivative(time, relative):
        return np.r_[
            relative[3:],
            relative_drift_acceleration(relative, env.n)
            + env._perturbation_acceleration(relative, time)
            + control,
        ]

    reference = solve_ivp(derivative, (0.0, env.dt), initial[:6], method="RK45", rtol=1e-10, atol=1e-12).y[:, -1]
    state, _ = env.step(control)
    error = np.max(np.abs(state[:6] - reference))
    # RK4 with 10 sub-steps over dt=1 s has fifth-order local truncation;
    # this 1e-6 threshold is a documented 25x margin over 3.977e-8 measured.
    assert error < 1e-6