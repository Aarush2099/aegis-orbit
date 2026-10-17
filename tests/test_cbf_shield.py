import numpy as np
import pytest
from scipy.optimize import minimize

from aegis_orbit.shields import ControlBarrierShield, SafetySupervisor
from aegis_orbit.shields.cbf_shield import _project_halfspace_box
from benchmarks.oracle_reference import load_config

_LEGACY_D_SAFE = float(load_config()["safety_geometry"]["d_safe_legacy_unused_m"])


def _shield(**kwargs):
    return ControlBarrierShield(
        d_safe=_LEGACY_D_SAFE,
        allow_unvalidated_d_safe=True,
        **kwargs,
    )


def test_legacy_config_radius_requires_explicit_override():
    legacy_radius = load_config()["safety_geometry"]["d_safe_legacy_unused_m"]

    with pytest.raises(ValueError, match="allow_unvalidated_d_safe"):
        ControlBarrierShield(d_safe=legacy_radius)

    shield = ControlBarrierShield(
        d_safe=legacy_radius,
        allow_unvalidated_d_safe=True,
    )
    assert shield.d_safe == legacy_radius


def test_projection_matches_reference():
    rng = np.random.default_rng(0)
    cases = []
    while len(cases) < 200:
        u_nom = rng.uniform(-1.0, 1.0, 3)
        a = rng.normal(size=3)
        u_max = 1.0
        nominal_dot = np.dot(a, np.clip(u_nom, -u_max, u_max))
        if nominal_dot <= 0.0:
            continue
        b = 0.5 * nominal_dot
        cases.append((u_nom, a, b, u_max))
    for u_nom, a, b, u_max in cases:
        reference_start = np.zeros(3)
        expected = minimize(
            lambda u: 0.5 * np.sum((u - u_nom) ** 2),
            reference_start,
            method="SLSQP",
            bounds=[(-u_max, u_max)] * 3,
            constraints={"type": "ineq", "fun": lambda u: b - np.dot(a, u)},
            options={"ftol": 1e-9, "maxiter": 500},
        )
        from aegis_orbit.shields.cbf_shield import _project_halfspace_box

        actual, feasible = _project_halfspace_box(u_nom, a, b, u_max)
        assert feasible and expected.success
        assert np.allclose(actual, expected.x, atol=1e-8)
        assert np.dot(a, actual) <= b + 1e-9
        assert np.all(np.abs(actual) <= u_max + 1e-12)


def test_projection_is_feasible_at_exact_box_boundary():
    normal = np.array([1.0, 0.0, 0.0])
    actual, feasible = _project_halfspace_box(np.zeros(3), normal, -1.0, 1.0)

    assert feasible
    assert np.array_equal(actual, np.array([-1.0, 0.0, 0.0]))
    assert np.dot(normal, actual) == -1.0

    nominal_at_bounds = np.array([-1.0, 1.0, 0.25])
    actual, feasible = _project_halfspace_box(nominal_at_bounds, normal, -1.0, 1.0)
    assert feasible
    assert np.array_equal(actual, nominal_at_bounds)


def test_projection_reports_small_analytic_correction():
    relative_position_m = 60.0
    d_safe_m = 50.0
    u_max = 1e-4
    h = relative_position_m**2 - d_safe_m**2
    b_target = 0.011999988
    alpha1 = 1.0
    alpha2 = b_target / (alpha1 * h)
    state = np.zeros(12, dtype=np.float64)
    state[0] = relative_position_m
    nominal = np.array([-u_max, 0.0, 0.0])
    shield = ControlBarrierShield(
        d_safe=d_safe_m,
        alpha1=alpha1,
        alpha2=alpha2,
        u_max=u_max,
        drift_fn=lambda _: np.zeros(3),
        allow_unvalidated_d_safe=True,
    )

    projected, projection_applied, feasible = shield.filter_action(state, nominal)
    expected_x = -b_target / (2.0 * relative_position_m)
    residual = -2.0 * relative_position_m * projected[0] - b_target

    assert feasible
    assert projection_applied
    assert np.isclose(projected[0], expected_x, atol=1e-11, rtol=0.0)
    assert abs(projected[0] - nominal[0]) < 1e-9
    assert abs(residual) <= 2e-9
    assert np.all(np.abs(projected) <= u_max)


def test_feasibility_classification_around_box_boundary():
    normal = np.array([1.0, 0.0, 0.0])
    just_inside, feasible_inside = _project_halfspace_box(
        np.zeros(3), normal, np.nextafter(-1.0, 0.0), 1.0
    )
    just_outside, feasible_outside = _project_halfspace_box(
        np.zeros(3), normal, np.nextafter(-1.0, -np.inf), 1.0
    )

    assert feasible_inside
    assert np.dot(normal, just_inside) <= np.nextafter(-1.0, 0.0)
    assert not feasible_outside
    assert np.array_equal(just_outside, np.array([-1.0, 0.0, 0.0]))


def test_projection_handles_zero_constraint_normal():
    nominal = np.array([2.0, -2.0, 0.5])
    zero_normal = np.zeros(3)

    actual, feasible = _project_halfspace_box(nominal, zero_normal, 0.0, 1.0)
    assert feasible
    assert np.array_equal(actual, np.array([1.0, -1.0, 0.5]))

    actual, feasible = _project_halfspace_box(nominal, zero_normal, -1e-12, 1.0)
    assert not feasible
    assert np.array_equal(actual, zero_normal)


def test_pass_through_when_safe():
    shield = _shield()
    state = np.array([200.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    nominal = np.array([1e-5, 0, 0], dtype=np.float64)
    safe, overridden, feasible = shield.filter_action(state, nominal)
    assert feasible and not overridden
    assert np.array_equal(safe, nominal)


def test_constraint_is_enforced_when_feasible():
    rng = np.random.default_rng(1)
    shield = _shield()
    for _ in range(50):
        r = rng.normal(size=3) * 200.0
        state = np.r_[r, rng.normal(size=3), np.zeros(6)]
        nominal = rng.uniform(-shield.u_max, shield.u_max, 3)
        safe, _, feasible = shield.filter_action(state, nominal)
        if feasible:
            v = state[3:6] - state[9:12]
            f = np.zeros(3)
            h = shield.barrier(r)
            h_dot = 2 * np.dot(r, v)
            b = 2 * np.dot(v, v) + 2 * np.dot(r, f) + shield.alpha1 * h_dot + shield.alpha2 * (h_dot + shield.alpha1 * h)
            assert np.dot(-2 * r, safe) <= b + 1e-9


def test_infeasible_branch_is_reported():
    shield = _shield(u_max=1e-12)
    state = np.zeros(12, dtype=np.float64)
    state[:3] = [1.0, 0.0, 0.0]
    safe, _, feasible = shield.filter_action(state, np.zeros(3))
    assert not feasible
    assert np.all(np.abs(safe) <= shield.u_max + 1e-24)


def test_drift_term_is_used():
    state = np.zeros(12, dtype=np.float64)
    state[:3] = [101.0, 0, 0]
    state[3] = -10.0
    nominal = np.zeros(3)
    first = _shield(u_max=10.0).filter_action(state, nominal)[0]
    second = _shield(u_max=10.0, drift_fn=lambda _: np.array([1.0, 0, 0])).filter_action(state, nominal)[0]
    assert not np.array_equal(first, second)


def test_backup_escape_direction_is_away_from_debris():
    shield = _shield(u_max=1e-12)
    state = np.zeros(12, dtype=np.float64)
    state[:3] = [50.0, 0, 0]
    applied, mode = SafetySupervisor(shield).act(state, np.zeros(3))
    assert mode == "backup" and applied[0] >= 0.0


@pytest.mark.slow
def test_closed_loop_smoke():
    shield = _shield()
    supervisor = SafetySupervisor(shield)
    state = np.zeros(12, dtype=np.float64)
    state[0] = 200.0
    state[3] = -0.5
    distances = []
    for _ in range(600):
        r = state[:3] - state[6:9]
        distances.append(np.linalg.norm(r))
        nominal = -shield.u_max * np.sign(r)
        acceleration, _ = supervisor.act(state, nominal)
        state[3:6] += acceleration
        state[:3] += state[3:6]
    infeasible = supervisor.counts["backup"]
    assert len(distances) == 600 and np.isfinite(min(distances))
    assert infeasible >= 0