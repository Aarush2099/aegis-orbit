"""Earth J2 acceleration and its local LVLH relative linearization.

The full acceleration follows the standard zonal-harmonic expression in
Montenbruck and Gill, *Satellite Orbits*, Springer (2000), Eq. 3.74. The
linearized model is the analytic Cartesian Jacobian of that expression,
transformed into the chief LVLH frame. This is equivalent to the local
first-order Schweighart-Sedwick relative model, while avoiding a copied
coefficient table whose convention depends on the LVLH definition.
"""

import numpy as np

from ...utils.orbital_math import MU_EARTH


def j2_acceleration_eci(
    r: np.ndarray, mu: float, j2: float, r_earth: float
) -> np.ndarray:
    """Return the J2-only acceleration in an inertial frame.

    Args:
        r: Inertial position in meters.
        mu: Gravitational parameter in cubic meters per second squared.
        j2: Dimensionless second zonal harmonic coefficient.
        r_earth: Reference Earth radius in meters.

    Returns:
        J2 acceleration in meters per second squared.

    The formula is the standard axisymmetric J2 term from Montenbruck and
    Gill (2000), Eq. 3.74. It omits two-body gravity, higher harmonics, and
    third-body effects; callers add those separately.
    """
    position = np.asarray(r, dtype=np.float64)
    radius_squared = float(np.dot(position, position))
    radius = np.sqrt(radius_squared)
    if radius == 0.0:
        raise ValueError("position must be nonzero")
    z_ratio_squared = position[2] ** 2 / radius_squared
    factor = 1.5 * j2 * mu * r_earth**2 / radius**5
    return np.asarray(
        [
            factor * position[0] * (5.0 * z_ratio_squared - 1.0),
            factor * position[1] * (5.0 * z_ratio_squared - 1.0),
            factor * position[2] * (5.0 * z_ratio_squared - 3.0),
        ],
        dtype=np.float64,
    )


def _j2_jacobian_eci(r: np.ndarray, mu: float, j2: float, r_earth: float) -> np.ndarray:
    position = np.asarray(r, dtype=np.float64)
    x, y, z = position
    radius_squared = float(np.dot(position, position))
    radius = np.sqrt(radius_squared)
    k = 1.5 * j2 * mu * r_earth**2
    common = 5.0 * z**2 / radius_squared
    diagonal = np.array([common - 1.0, common - 1.0, common - 3.0])
    gradient = np.empty((3, 3), dtype=np.float64)
    vector = np.array([x, y, z], dtype=np.float64)
    for row in range(3):
        for column in range(3):
            d_common = 10.0 * z / radius_squared if column == 2 else 0.0
            d_common -= 10.0 * z**2 * vector[column] / radius_squared**2
            d_vector = 1.0 if row == column else 0.0
            d_diagonal = d_common
            if row == 2:
                d_diagonal -= 0.0
            gradient[row, column] = k * (
                d_vector * diagonal[row] / radius**5
                + vector[row] * d_diagonal / radius**5
                - 5.0 * vector[row] * diagonal[row] * vector[column] / radius**7
            )
    return gradient


def _chief_lvLH_basis(inc: float, u_arg: float) -> np.ndarray:
    cu, su = np.cos(u_arg), np.sin(u_arg)
    ci, si = np.cos(inc), np.sin(inc)
    radial = np.array([cu, ci * su, si * su], dtype=np.float64)
    along_track = np.array([-su, ci * cu, si * cu], dtype=np.float64)
    cross_track = np.cross(radial, along_track)
    return np.vstack((radial, along_track, cross_track))


def j2_linearized_relative(
    x_rel: np.ndarray,
    n: float,
    j2: float,
    r_earth: float,
    a_chief: float,
    inc: float,
    u_arg: float,
) -> np.ndarray:
    """Return first-order J2 relative acceleration in chief LVLH axes.

    Args:
        x_rel: Relative [position, velocity] state in meters and meters per
            second; only its position components enter this position-gradient
            model.
        n: Chief mean motion in radians per second, retained for API parity.
        j2: Dimensionless second zonal harmonic coefficient.
        r_earth: Reference Earth radius in meters.
        a_chief: Chief circular-orbit radius in meters.
        inc: Chief inclination in radians.
        u_arg: Chief argument of latitude in radians.

    Returns:
        Linearized J2 relative acceleration in meters per second squared.

    The model is the local Schweighart-Sedwick first-order approximation,
    expressed as the analytic Jacobian of the standard full J2 field at the
    chief position. It assumes a circular chief, small deputy separation, and
    fixed local linearization over the call; it omits higher-order separation
    terms and all non-J2 perturbations.
    """
    del n
    relative = np.asarray(x_rel, dtype=np.float64)
    basis = _chief_lvLH_basis(inc, u_arg)
    chief_position = a_chief * basis[0]
    jacobian_eci = _j2_jacobian_eci(chief_position, MU_EARTH, j2, r_earth)
    jacobian_lvlh = basis @ jacobian_eci @ basis.T
    return np.asarray(jacobian_lvlh @ relative[:3], dtype=np.float64)