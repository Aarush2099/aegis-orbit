"""Differential atmospheric drag in a chief LVLH frame."""

import numpy as np


OMEGA_EARTH = 7.2921159e-5  # rad/s, IERS Earth rotation rate; verify epoch use.


def lvlh_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return orthonormal LVLH-to-inertial basis rows.

    Args:
        r: Inertial position in meters.
        v: Inertial velocity in meters per second.

    Returns:
        Three-by-three matrix whose rows are radial, along-track, and normal
        unit vectors in inertial coordinates.
    """
    radial = np.asarray(r, dtype=np.float64)
    radial /= np.linalg.norm(radial)
    normal = np.cross(radial, np.asarray(v, dtype=np.float64))
    normal /= np.linalg.norm(normal)
    along_track = np.cross(normal, radial)
    return np.vstack((radial, along_track, normal))


def drag_acceleration(
    v_inertial: np.ndarray,
    rho: float,
    ballistic_coeff: float,
    omega_earth: float,
    r: np.ndarray,
) -> np.ndarray:
    """Return atmospheric drag acceleration on one object.

    Args:
        v_inertial: Inertial velocity in meters per second.
        rho: Atmospheric density in kilograms per cubic meter.
        ballistic_coeff: ``Cd*A/m`` in square meters per kilogram.
        omega_earth: Earth rotation rate in radians per second.
        r: Inertial position in meters.

    Returns:
        Drag acceleration in meters per second squared.

    The atmosphere-relative velocity is ``v - omega cross r``. The rotation
    constant is the IERS conventional value 7.2921159e-5 rad/s; atmosphere
    winds, density gradients, and attitude-dependent coefficients are omitted.
    """
    relative_velocity = np.asarray(v_inertial, dtype=np.float64) - np.cross(
        np.array([0.0, 0.0, omega_earth], dtype=np.float64), r
    )
    speed = np.linalg.norm(relative_velocity)
    return np.asarray(
        -0.5 * rho * ballistic_coeff * speed * relative_velocity,
        dtype=np.float64,
    )


def differential_drag_lvlh(
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    r_deputy: np.ndarray,
    v_deputy: np.ndarray,
    rho: float,
    ballistic_coeff_chief: float,
    ballistic_coeff_deputy: float,
    omega_earth: float = OMEGA_EARTH,
) -> np.ndarray:
    """Return deputy-minus-chief drag in chief LVLH coordinates.

    Differential drag is generally small compared with J2 for similar
    altitudes and ballistic coefficients; its magnitude depends on the stated
    coefficient difference and is not assumed large.

    Args:
        r_chief, v_chief: Chief inertial position (m) and velocity (m/s).
        r_deputy, v_deputy: Deputy inertial position (m) and velocity (m/s).
        rho: Density in kilograms per cubic meter.
        ballistic_coeff_chief, ballistic_coeff_deputy: ``Cd*A/m`` in m^2/kg.
        omega_earth: Earth rotation rate in radians per second.

    Returns:
        Relative drag acceleration in chief LVLH axes, m/s^2.
    """
    chief = drag_acceleration(v_chief, rho, ballistic_coeff_chief, omega_earth, r_chief)
    deputy = drag_acceleration(v_deputy, rho, ballistic_coeff_deputy, omega_earth, r_deputy)
    return np.asarray(lvlh_basis(r_chief, v_chief) @ (deputy - chief), dtype=np.float64)