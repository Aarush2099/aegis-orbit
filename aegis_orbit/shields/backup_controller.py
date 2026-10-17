"""Fallback control for infeasible barrier-filter steps."""

import numpy as np

from .cbf_shield import ControlBarrierShield


def max_escape_thrust(r_rel: np.ndarray, u_max: float) -> np.ndarray:
    """Return componentwise bang-bang thrust away from the debris.

    Args:
        r_rel: Relative position, satellite minus debris, in meters.
        u_max: Per-axis acceleration limit in meters per second squared.

    Returns:
        Escape acceleration in meters per second squared.

    The componentwise sign pattern is not thrust along the unit vector: each
    nonzero axis independently receives its maximum magnitude. This is a
    heuristic fallback with no proof of safety.
    """
    return np.asarray(u_max * np.sign(np.asarray(r_rel, dtype=np.float64)), dtype=np.float64)


class SafetySupervisor:
    """Select nominal, shielded, or heuristic backup acceleration."""

    def __init__(self, shield: ControlBarrierShield):
        self.shield = shield
        self.counts = {"nominal": 0, "shielded": 0, "backup": 0}

    def act(self, state: np.ndarray, u_nom: np.ndarray) -> tuple[np.ndarray, str]:
        """Apply the shield and report the selected operating mode.

        Args:
            state: 12-state [r_sat, v_sat, r_deb, v_deb], in meters and meters
                per second.
            u_nom: Nominal acceleration in meters per second squared.

        Returns:
            Applied acceleration in meters per second squared and mode string.
        """
        u_safe, overridden, feasible = self.shield.filter_action(state, u_nom)
        if not feasible:
            r_rel = np.asarray(state)[0:3] - np.asarray(state)[6:9]
            u_applied = max_escape_thrust(r_rel, self.shield.u_max)
            mode = "backup"
        else:
            u_applied = u_safe
            mode = "shielded" if overridden else "nominal"
        self.counts[mode] += 1
        return u_applied, mode