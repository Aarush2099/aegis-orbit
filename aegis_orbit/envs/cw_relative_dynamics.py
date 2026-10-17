"""Clohessy-Wiltshire relative dynamics in an RTN-style frame."""

import numpy as np


def cw_state_transition(n: float, t: float) -> np.ndarray:
    """Return the 6-by-6 Clohessy-Wiltshire state transition matrix.

    Args:
        n: Mean motion in radians per second.
        t: Elapsed propagation time in seconds.

    Returns:
        State transition matrix for [x, y, z, vx, vy, vz], with positions in
        meters and velocities in meters per second.
    """
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    phi = np.zeros((6, 6), dtype=np.float64)
    phi[:3, :3] = np.array(
        [[4.0 - 3.0 * c, 0.0, 0.0], [6.0 * (s - nt), 1.0, 0.0], [0.0, 0.0, c]],
        dtype=np.float64,
    )
    phi[:3, 3:] = np.array(
        [[s / n, 2.0 * (1.0 - c) / n, 0.0],
         [2.0 * (c - 1.0) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
         [0.0, 0.0, s / n]],
        dtype=np.float64,
    )
    phi[3:, :3] = np.array(
        [[3.0 * n * s, 0.0, 0.0], [6.0 * n * (c - 1.0), 0.0, 0.0], [0.0, 0.0, -n * s]],
        dtype=np.float64,
    )
    phi[3:, 3:] = np.array(
        [[c, 2.0 * s, 0.0], [-2.0 * s, 4.0 * c - 3.0, 0.0], [0.0, 0.0, c]],
        dtype=np.float64,
    )
    return phi


def propagate_relative(x0: np.ndarray, n: float, t: float) -> np.ndarray:
    """Propagate a relative state under homogeneous CW dynamics.

    Args:
        x0: Initial [x, y, z, vx, vy, vz] state in meters and meters per
            second.
        n: Mean motion in radians per second.
        t: Elapsed time in seconds.

    Returns:
        Propagated six-state in meters and meters per second.
    """
    return np.asarray(cw_state_transition(n, t) @ np.asarray(x0, dtype=np.float64), dtype=np.float64)


def cw_control_input_matrix(n: float, t: float) -> np.ndarray:
    """Return the exact CW input matrix ``Gamma(t)``.

    Args:
        n: Mean motion in radians per second.
        t: Constant-control interval in seconds.

    Returns:
        Six-by-three matrix mapping constant acceleration in meters per second
        squared to a state increment in meters and meters per second.
    """
    # Long-double intermediates preserve precision in the cancellation-prone
    # in-track position term for short intervals; the public result remains
    # float64 as required by the environment contract.
    n_extended = np.longdouble(n)
    t_extended = np.longdouble(t)
    nt = n_extended * t_extended
    s, c = np.sin(nt), np.cos(nt)
    n_squared = n_extended**2
    return np.asarray(
        [
            [(1.0 - c) / n_squared, 2.0 * (nt - s) / n_squared, 0.0],
            [2.0 * (s - nt) / n_squared, 4.0 * (1.0 - c) / n_squared - 1.5 * t**2, 0.0],
            [0.0, 0.0, (1.0 - c) / n_squared],
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [2.0 * (c - 1.0) / n, 4.0 * s / n - 3.0 * t, 0.0],
            [0.0, 0.0, s / n],
        ],
        dtype=np.longdouble,
    ).astype(np.float64)


def relative_drift_acceleration(x_rel: np.ndarray, n: float) -> np.ndarray:
    """Return control-free CW relative acceleration.

    The sign convention is deputy/satellite relative to debris: ``r_rel`` is
    satellite minus debris in RTN coordinates. The result is the acceleration
    ``r_ddot = f(x) + u`` in meters per second squared, without control input.

    Args:
        x_rel: Relative [x, y, z, vx, vy, vz] state in meters and meters per
            second.
        n: Mean motion in radians per second.

    Returns:
        Relative drift acceleration in meters per second squared.
    """
    x_rel = np.asarray(x_rel, dtype=np.float64)
    x, _, z, vx, vy, vz = x_rel
    return np.asarray(
        [2.0 * n * vy + 3.0 * n**2 * x, -2.0 * n * vx, -n**2 * z],
        dtype=np.float64,
    )