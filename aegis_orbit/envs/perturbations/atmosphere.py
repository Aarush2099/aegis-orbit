"""Exponential density and stated solar-storm multiplier profiles."""

import numpy as np


def density_exponential(
    altitude_m: float, rho0: float, h0: float, scale_height: float
) -> float:
    """Return density from a local exponential atmosphere approximation.

    Args:
        altitude_m: Altitude in meters.
        rho0: Reference density in kilograms per cubic meter.
        h0: Reference altitude in meters.
        scale_height: Exponential scale height in meters.

    Returns:
        Density in kilograms per cubic meter.

    This is a local fit, not a measured storm model; composition and solar
    activity variations are omitted. Values should be sourced before use.
    """
    if scale_height <= 0.0 or rho0 < 0.0:
        raise ValueError("rho0 must be nonnegative and scale_height positive")
    return float(rho0 * np.exp(-(altitude_m - h0) / scale_height))


def storm_multiplier(
    t_s: float,
    onset_s: float,
    rise_s: float,
    peak_factor: float,
    decay_s: float,
) -> float:
    """Return a continuous fast-rise, slow-recovery density multiplier.

    Args:
        t_s: Elapsed time in seconds.
        onset_s: Storm onset time in seconds.
        rise_s: Linear-rise duration in seconds.
        peak_factor: Dimensionless multiplier at the peak.
        decay_s: Exponential recovery time constant in seconds.

    Returns:
        Dimensionless density multiplier.

    The piecewise shape states a fast rise and slow recovery for controlled
    experiments; it is not measured storm data and omits thermosphere physics.
    """
    if rise_s <= 0.0 or decay_s <= 0.0 or peak_factor < 1.0:
        raise ValueError("rise and decay must be positive and peak_factor >= 1")
    if t_s < onset_s:
        return 1.0
    elapsed = t_s - onset_s
    if elapsed <= rise_s:
        return float(1.0 + (peak_factor - 1.0) * elapsed / rise_s)
    return float(1.0 + (peak_factor - 1.0) * np.exp(-(elapsed - rise_s) / decay_s))