import numpy as np
from benchmarks.oracle_reference import load_config
from aegis_orbit.shields import ControlBarrierShield, SafetySupervisor, max_escape_thrust


_LEGACY_D_SAFE = float(load_config()["safety_geometry"]["d_safe_legacy_unused_m"])


def test_max_escape_thrust_uses_componentwise_sign():
    relative_position = np.array([2.0, -3.0, 0.0], dtype=np.float64)
    thrust = max_escape_thrust(relative_position, 1e-4)
    assert np.allclose(np.abs(thrust[:2]), 1e-4)
    assert np.sign(thrust[0]) == np.sign(relative_position[0])
    assert np.sign(thrust[1]) == np.sign(relative_position[1])
    assert thrust[2] == 0.0


def test_supervisor_counts_nominal_shielded_and_backup():
    shield = ControlBarrierShield(d_safe=_LEGACY_D_SAFE, u_max=1e-4, allow_unvalidated_d_safe=True)
    supervisor = SafetySupervisor(shield)

    nominal_state = np.zeros(12, dtype=np.float64)
    nominal_state[0] = 200.0
    _, nominal_mode = supervisor.act(nominal_state, np.zeros(3))

    shielded_state = nominal_state.copy()
    shielded_state[0] = 107.0
    shielded_state[3] = -0.075
    _, shielded_mode = supervisor.act(shielded_state, np.zeros(3))

    backup_shield = ControlBarrierShield(d_safe=_LEGACY_D_SAFE, u_max=1e-12, allow_unvalidated_d_safe=True)
    backup_supervisor = SafetySupervisor(backup_shield)
    backup_state = np.zeros(12, dtype=np.float64)
    backup_state[0] = 1.0
    _, backup_mode = backup_supervisor.act(backup_state, np.zeros(3))

    assert nominal_mode == "nominal"
    assert shielded_mode == "shielded"
    assert backup_mode == "backup"
    assert supervisor.counts == {"nominal": 1, "shielded": 1, "backup": 0}
    assert backup_supervisor.counts == {"nominal": 0, "shielded": 0, "backup": 1}