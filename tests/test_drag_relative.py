import numpy as np

from aegis_orbit.envs.perturbations.drag_relative import (
    differential_drag_lvlh,
    drag_acceleration,
    lvlh_basis,
)


def test_drag_opposes_atmosphere_relative_velocity():
    position = np.array([7e6, 0.0, 0.0])
    velocity = np.array([0.0, 7.5e3, 0.0])
    atmosphere_relative = velocity - np.cross([0.0, 0.0, 7.2921159e-5], position)
    acceleration = drag_acceleration(velocity, 1e-12, 2.0, 7.2921159e-5, position)
    assert np.dot(acceleration, atmosphere_relative) < 0.0


def test_drag_scales_with_density_and_ballistic_coefficient():
    r = np.array([7e6, 0.0, 0.0])
    v = np.array([0.0, 7.5e3, 0.0])
    base = drag_acceleration(v, 1e-12, 2.0, 7.2921159e-5, r)
    assert np.allclose(drag_acceleration(v, 2e-12, 2.0, 7.2921159e-5, r), 2.0 * base)
    assert np.allclose(drag_acceleration(v, 1e-12, 4.0, 7.2921159e-5, r), 2.0 * base)


def test_identical_objects_have_zero_differential_drag():
    r = np.array([7e6, 0.0, 0.0])
    v = np.array([0.0, 7.5e3, 0.0])
    result = differential_drag_lvlh(r, v, r, v, 1e-12, 2.0, 2.0)
    assert np.array_equal(result, np.zeros(3, dtype=np.float64))


def test_lvlh_rotation_is_orthonormal_and_preserves_norm():
    basis = lvlh_basis(np.array([7e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]))
    vector = np.array([1.0, 2.0, 3.0])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.norm(basis @ vector), np.linalg.norm(vector))