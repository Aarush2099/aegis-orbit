"""Constants and basic orbital calculations."""

import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2, Earth gravitational parameter.
R_EARTH = 6378137.0  # m, WGS-84 equatorial radius.
ALTITUDE_DEFAULT = 550e3  # m, default circular-orbit altitude.


def orbit_radius(altitude_m: float) -> float:
    """Return circular-orbit radius.

    Args:
        altitude_m: Altitude above the equatorial radius, in meters.

    Returns:
        Orbital radius in meters.
    """
    return float(R_EARTH + altitude_m)


def mean_motion(radius_m: float) -> float:
    """Return the circular-orbit mean motion.

    Args:
        radius_m: Circular-orbit radius in meters.

    Returns:
        Mean motion in radians per second.
    """
    return float(np.sqrt(MU_EARTH / radius_m**3))