import numpy as np
import pytest
from scipy.integrate import quad

from aegis_orbit.envs import cw_control_input_matrix, cw_state_transition, propagate_relative
from aegis_orbit.envs.continuous_conjunction import ContinuousConjunctionEnv


def _input_matrix_integrand(time: float, n: float, row: int, column: int) -> float:
    b = np.zeros((6, 3), dtype=np.float64)
    b[3:, :] = np.eye(3, dtype=np.float64)
    return float((cw_state_transition(n, time) @ b)[row, column])


def _make_env(dt: float = 1.0) -> ContinuousConjunctionEnv:
    return ContinuousConjunctionEnv(
        n=1.1e-3,
        dt=dt,
        horizon_steps=100,
        u_max=1e-4,
        d_safe=106.7,
        initial_offset_m=200.0,
        closing_speed_mps=0.5,
    )


def test_phi_reduces_to_identity_at_zero():
    assert np.allclose(cw_state_transition(1.1e-3, 0.0), np.eye(6), atol=1e-12)


def test_gamma_closed_form_matches_quadrature():
    n = 1.1e-3
    for dt in (1.0, 20.0, 60.0):
        gamma = cw_control_input_matrix(n, dt)
        reference = np.zeros((6, 3), dtype=np.float64)
        for row in range(6):
            for column in range(3):
                reference[row, column] = quad(
                    _input_matrix_integrand, 0.0, dt, args=(n, row, column),
                    epsabs=1e-13, epsrel=1e-13,
                )[0]
        # SciPy reports roundoff in this cancellation-prone entry at dt=1 s;
        # its absolute discrepancy is 1.32e-10, below this numerical floor.
        assert np.allclose(gamma, reference, rtol=1e-10, atol=2e-10)


def test_zero_thrust_matches_free_cw_propagation():
    env = _make_env()
    initial = env.reset(np.random.default_rng(0))
    expected = propagate_relative(initial[:6], env.n, env.dt)
    state, _ = env.step(np.zeros(3, dtype=np.float64))
    assert np.allclose(state[:6], expected, rtol=0.0, atol=1e-9)


def test_constant_thrust_matches_closed_form_forced_solution():
    env = _make_env()
    initial = env.reset(np.random.default_rng(1))
    control = np.array([3e-5, -2e-5, 1e-5], dtype=np.float64)
    expected = cw_state_transition(env.n, env.dt) @ initial[:6]
    expected += cw_control_input_matrix(env.n, env.dt) @ control
    state, _ = env.step(control)
    assert np.allclose(state[:6], expected, rtol=0.0, atol=1e-9)


def test_clipping_is_reported_and_enforced():
    env = _make_env()
    env.reset(np.random.default_rng(2))
    _, info = env.step(np.array([2e-4, -2e-4, 0.0], dtype=np.float64))
    assert info["clipped"] is True
    clipped_state = env.relative_state()

    env = _make_env()
    initial = env.reset(np.random.default_rng(2))
    _, info = env.step(np.array([1e-5, -1e-5, 0.0], dtype=np.float64))
    assert info["clipped"] is False
    expected = cw_state_transition(env.n, env.dt) @ initial[:6]
    expected += cw_control_input_matrix(env.n, env.dt) @ np.array([1e-5, -1e-5, 0.0])
    assert np.allclose(env.relative_state(), expected, atol=1e-9)
    assert not np.allclose(clipped_state, env.relative_state())


def test_reset_is_deterministic_and_jitter_is_bounded():
    env = _make_env()
    first = env.reset(np.random.default_rng(7))
    second = env.reset(np.random.default_rng(7))
    different = env.reset(np.random.default_rng(8))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.linalg.norm(first[1:3]) <= 0.01 * env.initial_offset_m


@pytest.mark.slow
def test_bounded_free_motion_over_ten_orbits():
    env = _make_env()
    env.reset(np.random.default_rng(0))
    # vy=-2*n*x cancels the CW secular in-track term, leaving x=x0*cos(nt).
    env._state[:6] = np.array([100.0, 0.0, 0.0, 0.0, -2.0 * env.n * 100.0, 0.0])
    positions = []
    for _ in range(10 * int(np.ceil(2.0 * np.pi / env.n / env.dt))):
        env.step(np.zeros(3, dtype=np.float64))
        positions.append(env.relative_state()[:3])
    assert np.max(np.abs(np.asarray(positions)[:, 0])) <= 100.0 + 1e-7


def test_substep_minimum_distance_has_documented_bound():
    env = _make_env(dt=1.0)
    env.reset(np.random.default_rng(0))
    env._state[:6] = np.array([0.25, 0.0, 0.0, -1.0, 0.0, 0.0])
    _, info = env.step(np.zeros(3, dtype=np.float64))
    dense_times = np.linspace(0.0, env.dt, 10001)
    dense_states = np.asarray([cw_state_transition(env.n, t) @ env._state[:6] for t in dense_times])
    true_minimum = float(np.min(np.linalg.norm(dense_states[:, :3], axis=1)))
    speed_bound = float(np.max(np.linalg.norm(dense_states[:, 3:], axis=1)))
    error_bound = speed_bound * env.dt / 40.0 + 1e-9
    assert info["min_distance_in_step"] - true_minimum <= error_bound


def test_optional_collision_trace_reports_first_substep_below_radius():
    env = _make_env(dt=1.0)
    env.reset(np.random.default_rng(0))
    env._state[:6] = np.array([0.25, 0.0, 0.0, -1.0, 0.0, 0.0])

    _, info = env.step(np.zeros(3, dtype=np.float64), collision_radius_m=0.24)

    assert info["first_collision_substep_index"] == 1
    assert info["first_collision_substep_time_s"] == pytest.approx(env.dt / 20.0)