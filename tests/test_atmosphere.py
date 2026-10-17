import numpy as np
import pytest

from aegis_orbit.envs.perturbations.atmosphere import density_exponential, storm_multiplier
from aegis_orbit.envs.perturbed_conjunction import altitude_above_surface_m
from benchmarks.oracle_reference import load_config


# Vallado, Fundamentals of Astrodynamics and Applications, Table 8-4
# (500 km row), as reproduced in arXiv:1610.02156; secondary source.
TABLE_RHO0_KG_M3 = 6.967e-13
TABLE_H0_ALTITUDE_M = 500000.0
TABLE_SCALE_HEIGHT_M = 63822.0
TABLE_EQUATORIAL_RADIUS_M = 6378137.0  # WGS-84 equatorial radius.


def test_density_is_reference_density_at_reference_altitude():
    assert density_exponential(
        TABLE_H0_ALTITUDE_M,
        TABLE_RHO0_KG_M3,
        TABLE_H0_ALTITUDE_M,
        TABLE_SCALE_HEIGHT_M,
    ) == TABLE_RHO0_KG_M3


def test_density_one_scale_height_above_reference_is_rho0_over_e():
    density = density_exponential(
        TABLE_H0_ALTITUDE_M + TABLE_SCALE_HEIGHT_M,
        TABLE_RHO0_KG_M3,
        TABLE_H0_ALTITUDE_M,
        TABLE_SCALE_HEIGHT_M,
    )
    assert np.isclose(density, TABLE_RHO0_KG_M3 / np.e, rtol=1e-12, atol=0.0)


def test_config_uses_sourced_atmosphere_table_values():
    atmosphere = load_config()["perturbations"]
    assert atmosphere["rho0"] == TABLE_RHO0_KG_M3
    assert atmosphere["h0"] == TABLE_H0_ALTITUDE_M
    assert atmosphere["scale_height"] == TABLE_SCALE_HEIGHT_M


def test_environment_density_uses_altitude_above_surface_at_550_km():
    # WGS-84 equatorial radius and the 550 km scenario altitude are used to
    # construct an Earth-centred radius; the source formula uses surface altitude.
    surface_altitude_m = 550000.0
    chief_radius_m = TABLE_EQUATORIAL_RADIUS_M + surface_altitude_m
    altitude_m = altitude_above_surface_m(np.array([chief_radius_m, 0.0, 0.0]))
    density = density_exponential(
        altitude_m,
        TABLE_RHO0_KG_M3,
        TABLE_H0_ALTITUDE_M,
        TABLE_SCALE_HEIGHT_M,
    )
    expected = TABLE_RHO0_KG_M3 * np.exp(
        -(550000.0 - 500000.0) / 63822.0
    )
    assert np.isclose(density, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("chief_coefficient", "deputy_coefficient", "missing_key"),
    [
        (None, object(), "chief_ballistic_coeff_m2_per_kg"),
        (object(), None, "deputy_ballistic_coeff_m2_per_kg"),
    ],
)
def test_drag_requires_each_ballistic_coefficient(
    chief_coefficient, deputy_coefficient, missing_key
):
    from aegis_orbit.envs import PerturbedConjunctionEnv

    with pytest.raises(ValueError, match=missing_key):
        PerturbedConjunctionEnv(
            n=1.1e-3,
            dt=1.0,
            horizon_steps=1,
            u_max=1e-4,
            d_safe=67.0,
            initial_offset_m=67.0,
            closing_speed_mps=0.5,
            drag_enabled=True,
            ballistic_coeff_chief=chief_coefficient,
            ballistic_coeff_deputy=deputy_coefficient,
            allow_unvalidated_d_safe=True,
        )


def test_density_reference_and_monotonicity():
    values = [density_exponential(height, 2e-12, 550e3, 60e3) for height in (500e3, 550e3, 600e3)]
    assert values[1] == 2e-12
    assert values[0] > values[1] > values[2]


def test_storm_profile_endpoints_and_recovery():
    kwargs = {"onset_s": 10.0, "rise_s": 20.0, "peak_factor": 2.0, "decay_s": 30.0}
    assert storm_multiplier(9.99, **kwargs) == 1.0
    assert storm_multiplier(30.0, **kwargs) == 2.0
    assert abs(storm_multiplier(30.0 + 5.0 * 30.0, **kwargs) - 1.0) < 0.01


def test_storm_profile_is_continuous_at_transition():
    kwargs = {"onset_s": 10.0, "rise_s": 20.0, "peak_factor": 2.0, "decay_s": 30.0}
    before = storm_multiplier(30.0 - 1e-9, **kwargs)
    after = storm_multiplier(30.0 + 1e-9, **kwargs)
    assert np.isclose(before, after, atol=1e-9)