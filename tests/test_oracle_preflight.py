import copy

import numpy as np
import pytest

from benchmarks import oracle_reference
from benchmarks.oracle_reference import load_config
import benchmarks.run_unshielded_benchmark as benchmark


class _PassiveProbeEnvironment:
    horizon_steps = 1

    def __init__(self, seed):
        self.seed = seed

    def reset(self, rng):
        return np.zeros(12)

    def step(self, action):
        collided = self.seed < 69
        return np.zeros(12), {
            "min_distance_in_step": 9.0 if collided else 20.0,
            "distance": 9.0 if collided else 20.0,
        }


def _mock_probe(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "make_fixed_environment",
        lambda config, seed, offset: _PassiveProbeEnvironment(
            seed - int(config["oracle_preflight"]["evaluation_seeds"]["start"])
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_encounter",
        lambda seed, config, hard: (True, hard + 0.5),
    )


def test_make_fixed_environment_uses_fixed_offset_without_range(monkeypatch):
    constructor_kwargs = {}

    def capture_constructor(**kwargs):
        constructor_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "ContinuousConjunctionEnv", capture_constructor)
    config = load_config()

    benchmark.make_fixed_environment(config, seed=10000, offset_m=67.0)

    assert constructor_kwargs["initial_offset_m"] == 67.0
    assert constructor_kwargs["initial_offset_range_m"] is None


@pytest.mark.slow
def test_oracle_preflight_passes_real_config(monkeypatch):
    config = load_config()
    original_band = copy.deepcopy(config["environment"]["initial_offset_m"])
    passive_initial_states = []
    oracle_initial_states = []
    passive_constructor_kwargs = []
    oracle_constructor_kwargs = []
    oracle_seeds = []
    environment_class = benchmark.ContinuousConjunctionEnv
    oracle_environment_class = oracle_reference.ContinuousConjunctionEnv

    def recording_constructor(environment_class, recorded_states, constructor_kwargs):
        def construct(**kwargs):
            constructor_kwargs.append(kwargs.copy())
            environment = environment_class(**kwargs)
            reset = environment.reset

            def recording_reset(rng):
                state = reset(rng)
                recorded_states.append(state.copy())
                return state

            environment.reset = recording_reset
            return environment

        return construct

    monkeypatch.setattr(
        benchmark,
        "ContinuousConjunctionEnv",
        recording_constructor(
            environment_class,
            passive_initial_states,
            passive_constructor_kwargs,
        ),
    )
    monkeypatch.setattr(
        oracle_reference,
        "ContinuousConjunctionEnv",
        recording_constructor(
            oracle_environment_class,
            oracle_initial_states,
            oracle_constructor_kwargs,
        ),
    )
    evaluate_encounter = benchmark.evaluate_encounter

    def record_oracle_seed(seed, checked_config, hard_body_radius_m):
        oracle_seeds.append(seed)
        return evaluate_encounter(seed, checked_config, hard_body_radius_m)

    monkeypatch.setattr(benchmark, "evaluate_encounter", record_oracle_seed)

    observed = benchmark.oracle_preflight(config)

    assert observed["passive_collision_count"] == 69
    assert observed["minimum_margin_m"] >= 0.25
    assert config["environment"]["initial_offset_m"] == original_band
    assert passive_constructor_kwargs
    assert all(kwargs["initial_offset_m"] == 67.0 for kwargs in passive_constructor_kwargs)
    assert all(kwargs["initial_offset_range_m"] is None for kwargs in passive_constructor_kwargs)
    assert oracle_constructor_kwargs[0]["initial_offset_m"] == 67.0
    assert oracle_seeds[0] == 10000
    assert np.array_equal(passive_initial_states[0], oracle_initial_states[0])


@pytest.mark.slow
def test_oracle_preflight_rejects_margin_below_configured_minimum(monkeypatch):
    _mock_probe(monkeypatch)
    config = load_config()
    config["oracle_preflight"]["min_margin_m"] = 1000.0

    with pytest.raises(RuntimeError, match="minimum_margin_m"):
        benchmark.oracle_preflight(config)


@pytest.mark.slow
def test_oracle_preflight_rejects_wrong_passive_collision_count(monkeypatch):
    _mock_probe(monkeypatch)
    config = load_config()
    config["oracle_preflight"]["passive_collision_count"] = 0

    with pytest.raises(RuntimeError, match="observed_passive_collisions=69"):
        benchmark.oracle_preflight(config)
