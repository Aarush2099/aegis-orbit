import numpy as np

from aegis_orbit.utils import R_EARTH, mean_motion, orbit_radius


def test_orbit_radius():
    assert np.isclose(orbit_radius(550e3), R_EARTH + 550e3, atol=1e-6)


def test_mean_motion_at_550_km():
    assert 1.09e-3 < mean_motion(orbit_radius(550e3)) < 1.12e-3