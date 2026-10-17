from benchmarks.oracle_reference import load_config
from benchmarks.run_unshielded_benchmark import make_environment


def test_environment_uses_explicit_nested_parameters():
    config = load_config()
    environment = make_environment(config, 0)
    assert environment.d_safe == config["safety_geometry"]["d_safe_legacy_unused_m"]
    assert environment.initial_offset_range_m == (67.0, 67.5)
