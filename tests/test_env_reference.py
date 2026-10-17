import numpy as np
from scipy.integrate import solve_ivp

from aegis_orbit.envs import cw_state_transition
from aegis_orbit.envs.continuous_conjunction import ContinuousConjunctionEnv


def test_environment_matches_independent_linear_ode_reference():
    """Check discretization against an ODE solver, not physical reality."""
    n = 1.1e-3
    dt = 1.0
    env = ContinuousConjunctionEnv(n, dt, 60, 1e-4, 106.7, 200.0, 0.5)
    initial = env.reset(np.random.default_rng(0))
    rng = np.random.default_rng(0)
    controls = rng.uniform(-1e-4, 1e-4, size=(60, 3)).astype(np.float64)
    a = np.zeros((6, 6), dtype=np.float64)
    a[:3, 3:] = np.eye(3, dtype=np.float64)
    a[3:, :3] = np.diag([3.0 * n**2, 0.0, -n**2])
    a[3, 4] = 2.0 * n
    a[4, 3] = -2.0 * n
    b = np.zeros((6, 3), dtype=np.float64)
    b[3:, :] = np.eye(3, dtype=np.float64)
    reference = initial[:6].copy()
    max_position_error = 0.0
    max_velocity_error = 0.0
    for control in controls:
        result = solve_ivp(
            lambda time, state: a @ state + b @ control,
            (0.0, dt),
            reference,
            method="RK45",
            rtol=1e-11,
            atol=1e-12,
        )
        reference = result.y[:, -1]
        state, _ = env.step(control)
        max_position_error = max(max_position_error, float(np.max(np.abs(state[:3] - reference[:3]))))
        max_velocity_error = max(max_velocity_error, float(np.max(np.abs(state[3:6] - reference[3:]))))
    assert max_position_error < 1e-6
    assert max_velocity_error < 1e-9