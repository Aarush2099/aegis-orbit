"""Continuous conjunction environment with switchable perturbations."""

import numpy as np

from ..shields import ControlBarrierShield
from ..utils.orbital_math import MU_EARTH, R_EARTH
from .continuous_conjunction import ContinuousConjunctionEnv
from .cw_relative_dynamics import (
    cw_control_input_matrix,
    cw_state_transition,
    relative_drift_acceleration,
)
from .perturbations.atmosphere import density_exponential, storm_multiplier
from .perturbations.drag_relative import differential_drag_lvlh
from .perturbations.j2_relative import (
    _chief_lvLH_basis,
    j2_acceleration_eci,
    j2_linearized_relative,
)


def altitude_above_surface_m(position: np.ndarray) -> float:
    """Convert an Earth-centred position radius to altitude above the surface."""
    return float(np.linalg.norm(np.asarray(position, dtype=np.float64)) - R_EARTH)


class PerturbedConjunctionEnv:
    """Wrap the CW base environment with J2 and drag perturbations.

    The CW and control portions are exact for each piecewise-constant step.
    Perturbations are integrated by ten fixed RK4 substeps and added as a
    splitting correction. This is a numerical research model with circular
    chief assumptions; it omits higher harmonics, winds, density estimation,
    actuator dynamics, and third-body forces.
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
        *,
        initial_offset_range_m: tuple[float, float] | None = None,
        j2_enabled: bool = False,
        drag_enabled: bool = False,
        storm: dict[str, float] | None = None,
        truth_model: str = "cw",
        shield_model: str = "cw",
        allow_unvalidated_d_safe: bool = False,
        ballistic_coeff_chief: float | None = None,
        ballistic_coeff_deputy: float | None = None,
        a_chief: float | None = None,
        inc: float = 0.0,
        u_arg: float = 0.0,
        rho0: float = 6.967e-13,  # Vallado, Fundamentals of Astrodynamics and Applications, Table 8-4 (500 km row), as reproduced in arXiv:1610.02156; secondary source, primary verification pending.
        h0: float = 500000.0,  # Same secondary source; reference altitude is above Earth's surface.
        scale_height: float = 63822.0,  # Same secondary source; primary verification pending.
    ) -> None:
        """Initialize the perturbed plain-Python environment.

        Args:
            n, dt, horizon_steps, u_max, d_safe, initial_offset_m,
                closing_speed_mps: Same SI-valued CW parameters as the base
                environment.
            j2_enabled: Enable nonlinear differential J2 in truth dynamics.
            drag_enabled: Enable differential atmospheric drag.
            storm: Storm-profile keyword parameters, or ``None`` for multiplier
                one. Profile parameters use seconds and a dimensionless factor.
            truth_model: ``"cw"`` or ``"cw_j2"`` for the true model label.
            shield_model: ``"cw"`` or ``"cw_j2"`` for the nominal shield drift.
            allow_unvalidated_d_safe: Explicitly opt in to a radius without a
                validated uncertainty derivation.
            ballistic_coeff_chief, ballistic_coeff_deputy: Required ``Cd*A/m``
                values in square meters per kilogram when drag is enabled.
            a_chief: Chief radius in meters; defaults to the radius implied by
                ``n`` and Earth mu.
            inc, u_arg: Chief inclination and initial argument of latitude in
                radians.
            rho0, h0, scale_height: Exponential atmosphere parameters in
                kg/m^3 and meters. The defaults use the sourced-secondary
                Vallado Table 8-4 500 km row reproduced in arXiv:1610.02156;
                primary verification against the book is pending.
        """
        if truth_model not in {"cw", "cw_j2"} or shield_model not in {"cw", "cw_j2"}:
            raise ValueError("truth_model and shield_model must be cw or cw_j2")
        if drag_enabled and ballistic_coeff_chief is None:
            raise ValueError(
                "chief_ballistic_coeff_m2_per_kg is required when drag is enabled"
            )
        if drag_enabled and ballistic_coeff_deputy is None:
            raise ValueError(
                "deputy_ballistic_coeff_m2_per_kg is required when drag is enabled"
            )
        self.base = ContinuousConjunctionEnv(
            n=n,
            dt=dt,
            horizon_steps=horizon_steps,
            u_max=u_max,
            d_safe=d_safe,
            initial_offset_m=initial_offset_m,
            closing_speed_mps=closing_speed_mps,
            initial_offset_range_m=initial_offset_range_m,
        )
        self.n = float(n)
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.j2_enabled = bool(j2_enabled)
        self.drag_enabled = bool(drag_enabled)
        self.storm = None if storm is None else dict(storm)
        self.truth_model = truth_model
        self.shield_model = shield_model
        self.ballistic_coeff_chief = ballistic_coeff_chief
        self.ballistic_coeff_deputy = ballistic_coeff_deputy
        self.a_chief = float(a_chief if a_chief is not None else (MU_EARTH / n**2) ** (1.0 / 3.0))
        self.inc = float(inc)
        self.u_arg = float(u_arg)
        self.rho0 = float(rho0)
        self.h0 = float(h0)
        self.scale_height = float(scale_height)
        self._state = np.zeros(12, dtype=np.float64)
        self._has_reset = False
        self._time = 0.0
        self._steps = 0
        self.shield = ControlBarrierShield(
            d_safe=d_safe,
            u_max=u_max,
            drift_fn=self._shield_drift,
            allow_unvalidated_d_safe=allow_unvalidated_d_safe,
        )

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Reset the wrapped CW encounter using an explicit generator.

        Args:
            rng: Explicit NumPy generator for the base encounter jitter.

        Returns:
            Initial 12-state in meters and meters per second.
        """
        self._state = self.base.reset(rng)
        self._time = 0.0
        self._steps = 0
        self._has_reset = True
        return self._state.copy()

    def _chief_state(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis = _chief_lvLH_basis(self.inc, self.u_arg + self.n * time_s)
        position = self.a_chief * basis[0]
        velocity = self.a_chief * self.n * basis[1]
        return position, velocity, basis

    def _perturbation_acceleration(self, relative: np.ndarray, time_s: float) -> np.ndarray:
        chief_position, chief_velocity, basis = self._chief_state(time_s)
        relative_position = relative[:3]
        deputy_position = chief_position + basis.T @ relative_position
        deputy_velocity = chief_velocity + basis.T @ relative[3:]
        acceleration = np.zeros(3, dtype=np.float64)
        if self.j2_enabled and self.truth_model == "cw_j2":
            chief_j2 = j2_acceleration_eci(chief_position, MU_EARTH, 1.08263e-3, R_EARTH)
            deputy_j2 = j2_acceleration_eci(deputy_position, MU_EARTH, 1.08263e-3, R_EARTH)
            acceleration += basis @ (deputy_j2 - chief_j2)
        if self.drag_enabled:
            multiplier = 1.0 if self.storm is None else storm_multiplier(time_s, **self.storm)
            rho = density_exponential(
                altitude_above_surface_m(chief_position),
                self.rho0,
                self.h0,
                self.scale_height,
            ) * multiplier
            acceleration += differential_drag_lvlh(
                chief_position,
                chief_velocity,
                deputy_position,
                deputy_velocity,
                rho,
                float(self.ballistic_coeff_chief),
                float(self.ballistic_coeff_deputy),
            )
        return np.asarray(acceleration, dtype=np.float64)

    def _shield_drift(self, state: np.ndarray) -> np.ndarray:
        relative = np.asarray(state, dtype=np.float64)[0:6]
        drift = relative_drift_acceleration(relative, self.n)
        if self.shield_model == "cw_j2":
            drift += j2_linearized_relative(
                relative, self.n, 1.08263e-3, R_EARTH,
                self.a_chief, self.inc, self.u_arg + self.n * self._time,
            )
        return drift

    def step(self, u_nom: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """Advance one clipped control interval with perturbation splitting.

        Args:
            u_nom: Requested relative acceleration in meters per second squared.

        Returns:
            New 12-state and perturbation/model metadata.
        """
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        requested = np.asarray(u_nom, dtype=np.float64)
        applied = np.clip(requested, -self.u_max, self.u_max)
        clipped = bool(not np.array_equal(requested, applied))
        relative = self._state[:6].copy()
        cw_next = cw_state_transition(self.n, self.dt) @ relative
        cw_next += cw_control_input_matrix(self.n, self.dt) @ applied
        correction = np.zeros(6, dtype=np.float64)
        minimum_distance = float(np.linalg.norm(relative[:3]))
        sub_dt = self.dt / 10.0  # s, fixed perturbation RK4 interval.
        for index in range(10):
            start = self._time + index * sub_dt

            def stage_relative(delta: np.ndarray, local_time: float) -> np.ndarray:
                cw_local = cw_state_transition(self.n, local_time - self._time) @ relative
                cw_local += cw_control_input_matrix(self.n, local_time - self._time) @ applied
                return cw_local + delta

            def derivative(delta: np.ndarray, local_time: float) -> np.ndarray:
                output = np.zeros(6, dtype=np.float64)
                output[:3] = delta[3:]
                output[3:] = self._perturbation_acceleration(
                    stage_relative(delta, local_time), local_time
                )
                return output

            minimum_distance = min(
                minimum_distance,
                float(np.linalg.norm(stage_relative(correction, start)[:3])),
            )
            k1 = derivative(correction, start)
            midpoint = correction + 0.5 * sub_dt * k1
            minimum_distance = min(
                minimum_distance,
                float(
                    np.linalg.norm(
                        stage_relative(midpoint, start + 0.5 * sub_dt)[:3]
                    )
                ),
            )
            k2 = derivative(midpoint, start + 0.5 * sub_dt)
            k3 = derivative(
                correction + 0.5 * sub_dt * k2, start + 0.5 * sub_dt
            )
            k4 = derivative(correction + sub_dt * k3, start + sub_dt)
            correction += sub_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        final_relative = cw_next + correction
        minimum_distance = min(
            minimum_distance, float(np.linalg.norm(final_relative[:3]))
        )
        self._state[:6] = final_relative
        self._time += self.dt
        self._steps += 1
        perturbation = self._perturbation_acceleration(final_relative, self._time)
        info: dict[str, object] = {
            "clipped": clipped,
            "perturbation_accel": perturbation,
            "storm_multiplier": 1.0 if self.storm is None else storm_multiplier(self._time, **self.storm),
            "truth_model": self.truth_model,
            "shield_model": self.shield_model,
            "min_distance_in_step": minimum_distance,
        }
        return self._state.copy(), info

    def relative_state(self) -> np.ndarray:
        """Return the current relative state in meters and meters per second."""
        return self._state[:6].copy()

    @property
    def time(self) -> float:
        """Return elapsed time in seconds."""
        return float(self._time)

    @property
    def done(self) -> bool:
        """Return whether the configured horizon has been reached."""
        return bool(self._steps >= self.base.horizon_steps)