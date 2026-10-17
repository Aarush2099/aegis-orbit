"""
AegisOrbit safety filter: second-order control barrier function (CBF) shield.

Per control step, solve exactly:
    min_u  0.5 * ||u - u_nom||^2
    s.t.   a . u <= b         (CBF constraint, a = -2 r)
           |u_i| <= u_max     (thrust box)
"""
import numpy as np


def _project_halfspace_box(u_nom, a, b, u_max, max_iter=200, tol=1e-12):
    """Exact projection onto {u : a.u <= b, |u_i| <= u_max}. Returns (u, feasible)."""
    u0 = np.clip(u_nom, -u_max, u_max)
    if np.dot(a, u0) <= b:
        return u0, True

    if -u_max * np.sum(np.abs(a)) > b:
        # No admissible thrust satisfies the barrier this step.
        return -u_max * np.sign(a), False

    lo, hi = 0.0, 1.0
    while np.dot(a, np.clip(u_nom - hi * a, -u_max, u_max)) > b:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if np.dot(a, np.clip(u_nom - mid * a, -u_max, u_max)) > b:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return np.clip(u_nom - hi * a, -u_max, u_max), True


class ControlBarrierShield:
    """
    Barrier:          h   = ||r||^2 - D^2          (safe set: h >= 0)
    Auxiliary:        psi = h_dot + alpha1 * h
    Invariance:       psi_dot + alpha2 * psi >= 0
    """

    def __init__(
        self,
        d_safe,
        alpha1=0.5,
        alpha2=0.5,
        u_max=1e-4,
        drift_fn=None,
        *,
        allow_unvalidated_d_safe=False,
    ):
        if d_safe is None:
            raise ValueError("d_safe must be explicitly supplied")
        if not allow_unvalidated_d_safe:
            raise ValueError(
                "d_safe has no validated derived value; explicit "
                "allow_unvalidated_d_safe=True is required for an override"
            )
        self.d_safe = float(d_safe)
        self.alpha1 = alpha1  # 1/s; barrier gain.
        self.alpha2 = alpha2  # 1/s; barrier gain.
        self.u_max = u_max  # m/s^2; replace with a stated thruster model value.
        self.drift_fn = drift_fn

    def barrier(self, r):
        """Return the barrier value for a relative position.

        Args:
            r: Relative position in meters.

        Returns:
            Barrier value in square meters.
        """
        return float(np.dot(r, r) - self.d_safe**2)

    def filter_action(self, state, u_nom):
        """Project a nominal acceleration through the CBF constraint.

        Args:
            state: 12-state [r_sat, v_sat, r_deb, v_deb], in meters and meters
                per second.
            u_nom: Nominal acceleration in meters per second squared.

        Returns:
            Tuple of safe acceleration, override flag, and QP feasibility flag.
        """
        state = np.asarray(state, float)
        u_nom = np.asarray(u_nom, float)
        r = state[0:3] - state[6:9]
        v = state[3:6] - state[9:12]
        f = np.zeros(3) if self.drift_fn is None else np.asarray(self.drift_fn(state), float)

        h = self.barrier(r)
        h_dot = 2.0 * np.dot(r, v)
        psi = h_dot + self.alpha1 * h

        a = -2.0 * r
        b = 2.0 * np.dot(v, v) + 2.0 * np.dot(r, f) + self.alpha1 * h_dot + self.alpha2 * psi

        u_safe, feasible = _project_halfspace_box(u_nom, a, b, self.u_max)
        overridden = not np.array_equal(u_safe, u_nom)
        return u_safe, overridden, feasible