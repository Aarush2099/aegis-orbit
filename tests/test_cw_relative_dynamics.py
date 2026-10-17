import numpy as np

from aegis_orbit.envs import cw_state_transition, propagate_relative, relative_drift_acceleration
from aegis_orbit.utils import mean_motion, orbit_radius


def test_transition_at_zero_is_identity():
    assert np.allclose(cw_state_transition(1.1e-3, 0.0), np.eye(6), atol=1e-12)


def test_drift_matches_finite_difference():
    n = 1.1e-3
    x0 = np.array([20.0, -5.0, 3.0, 0.2, -0.4, 0.1], dtype=np.float64)
    dt = 1e-5
    derivative = (propagate_relative(x0, n, dt)[3:] - x0[3:]) / dt
    expected = relative_drift_acceleration(x0, n)
    assert np.allclose(derivative, expected, rtol=1e-6, atol=1e-9)


def test_radial_offset_is_finite_and_bounded_for_one_orbit():
    n = mean_motion(orbit_radius(550e3))
    x0 = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    samples = [propagate_relative(x0, n, t) for t in np.linspace(0.0, 2.0 * np.pi / n, 100)]
    values = np.asarray(samples)
    assert np.isfinite(values).all()
    # The radial zero-velocity case has a bounded finite excursion over one
    # orbit, including the standard CW in-track secular term.
    assert np.max(np.abs(values[:, :3])) < 5_000.0