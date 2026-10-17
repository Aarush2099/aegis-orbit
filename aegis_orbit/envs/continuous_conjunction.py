"""Plain-Python continuous-thrust environment for linearized CW encounters."""

import numpy as np

from .cw_relative_dynamics import (
    cw_control_input_matrix,
    cw_state_transition,
)


class ContinuousConjunctionEnv:
    """Simulate a linearized, unperturbed relative-motion encounter.

    This is a linearized relative-motion model about a circular reference orbit,
    not an operational environment. The minimum-distance report
    samples 20 equal sub-intervals; under a known speed bound ``V`` its downward
    sampling error is at most ``V * dt / 40`` because a closest point is within
    half a sample interval of a sample.
    """

    def __init__(
        self,
        n: float,
        dt: float,
        horizon_steps: int,
        u_max: float,
        d_safe: float,
        initial_offset_m: float,
        closing_speed_mps: float,
        initial_offset_range_m: tuple[float, float] | None = None,
    ) -> None:
        """Initialize a continuous-thrust CW encounter.

        Args:
            n: Mean motion in radians per second.
            dt: Control interval in seconds.
            horizon_steps: Episode length in control steps.
            u_max: Per-axis acceleration limit in meters per second squared.
            d_safe: Reporting safety radius in meters.
            initial_offset_m: Initial radial separation in meters.
            closing_speed_mps: Initial closing speed in meters per second.
        """
        if n <= 0.0 or dt <= 0.0 or horizon_steps <= 0 or u_max <= 0.0:
            raise ValueError("n, dt, horizon_steps, and u_max must be positive")
        if d_safe < 0.0 or initial_offset_m <= 0.0 or closing_speed_mps < 0.0:
            raise ValueError("distance parameters must be nonnegative and offset positive")
        self.n = float(n)
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.u_max = float(u_max)
        self.d_safe = float(d_safe)
        self.initial_offset_m = float(initial_offset_m)
        self.initial_offset_range_m = initial_offset_range_m
        self.closing_speed_mps = float(closing_speed_mps)
        self._state = np.zeros(12, dtype=np.float64)
        self._time = 0.0
        self._steps = 0
        self._has_reset = False

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Reset to a seeded radial encounter with bounded lateral jitter.

        Args:
            rng: Explicit NumPy generator supplying lateral-jitter randomness.

        Returns:
            Initial 12-state in meters and meters per second.
        """
        if self.initial_offset_range_m is None:
            offset = self.initial_offset_m
        else:
            low, high = self.initial_offset_range_m
            offset = float(rng.uniform(low, high))
        jitter_limit = 0.01 * offset  # m, maximum lateral jitter.
        radius = float(rng.uniform(0.0, jitter_limit))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))  # rad, lateral jitter angle.
        relative_position = np.array(
            [offset, radius * np.cos(angle), radius * np.sin(angle)],
            dtype=np.float64,
        )
        relative_velocity = np.array(
            [-self.closing_speed_mps, 0.0, 0.0], dtype=np.float64
        )
        self._state = np.zeros(12, dtype=np.float64)
        self._state[:3] = relative_position
        self._state[3:6] = relative_velocity
        self._time = 0.0
        self._steps = 0
        self._has_reset = True
        return self._state.copy()

    def step(
        self,
        u_nom: np.ndarray,
        *,
        collision_radius_m: float | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply one clipped constant-acceleration control interval.

        Args:
            u_nom: Requested relative acceleration in meters per second squared.
            collision_radius_m: Optional threshold for recording the first
                sampled sub-step below the collision radius.

        Returns:
            New 12-state and info containing clipping, sampled minimum distance,
            end distance, and elapsed time. The minimum uses 20 sub-steps and
            therefore has the speed-bound sampling error described by the class.
        """
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        requested = np.asarray(u_nom, dtype=np.float64)
        if requested.shape != (3,):
            raise ValueError("u_nom must have shape (3,)")
        applied = np.clip(requested, -self.u_max, self.u_max)
        clipped = bool(not np.array_equal(applied, requested))
        relative = self._state[[0, 1, 2, 3, 4, 5]]
        times = np.linspace(0.0, self.dt, 21, dtype=np.float64)
        states = np.asarray(
            [cw_state_transition(self.n, sub_dt) @ relative + cw_control_input_matrix(self.n, sub_dt) @ applied
             for sub_dt in times],
            dtype=np.float64,
        )
        final_relative = states[-1]
        self._state[:3] = final_relative[:3]
        self._state[3:6] = final_relative[3:]
        self._time += self.dt
        self._steps += 1
        distances = np.linalg.norm(states[:, :3], axis=1)
        info: dict[str, object] = {
            "clipped": clipped,
            "min_distance_in_step": float(np.min(distances)),
            "distance": float(distances[-1]),
            "time": float(self._time),
        }
        if collision_radius_m is not None:
            if not np.isfinite(collision_radius_m) or collision_radius_m < 0.0:
                raise ValueError("collision_radius_m must be finite and nonnegative")
            below_radius = np.flatnonzero(distances < collision_radius_m)
            first_index = int(below_radius[0]) if len(below_radius) else None
            info["first_collision_substep_index"] = first_index
            info["first_collision_substep_time_s"] = (
                float(times[first_index]) if first_index is not None else None
            )
        return self._state.copy(), info

    def relative_state(self) -> np.ndarray:
        """Return the current relative state in meters and meters per second."""
        return self._state[:6].copy()

    @property
    def time(self) -> float:
        """Return elapsed episode time in seconds."""
        return float(self._time)

    @property
    def done(self) -> bool:
        """Return whether the configured control-step horizon is reached."""
        return bool(self._steps >= self.horizon_steps)