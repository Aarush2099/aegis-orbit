import benchmarks.run_unshielded_benchmark as benchmark


def test_passive_baseline_reports_one_progress_line_per_episode(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark,
        "run_episode",
        lambda *args, **kwargs: {"collided": False, "fuel_mps": 0.0},
    )
    monkeypatch.setattr(benchmark, "make_environment", lambda config, seed: object())

    rows = benchmark.run_baseline_episodes(
        "passive",
        benchmark.passive_policy,
        range(3),
        {"environment": {"u_max": 1.0}},
        10.0,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3
    assert len(lines) == 3
    assert all("passive episode" in line for line in lines)
    assert all(f"episode {index}/3" in line for index, line in enumerate(lines, start=1))
