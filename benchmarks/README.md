# Benchmark status

- Fast suite: passing in 30.62 seconds.
- Smoke run: a prior 6,000-step smoke passed the six artifact-integrity checks;
  it is diagnostic only, not a result. A current end-to-end smoke has not been
  run.
- Full benchmark: no completed full run.
- Oracle pre-flight: implemented and tested; the real config reports 69 passive
  collisions, 69 oracle-avoidable encounters, and a 0.426314 m minimum margin.
- Original six Step 2C gate definitions: not verified in this repository.

For an isolated smoke, make a temporary copy of `configs/sac_unshielded.yaml`
under `$env:TEMP`, set `training.total_env_steps` to 6000,
`training.validation_interval_steps` to 2000, and both evaluation and validation
seed counts to 2. Keep the oracle pre-flight configuration unchanged. Run:

```powershell
python -m benchmarks.run_unshielded_benchmark --config <temporary-config.yaml> --output-dir <temporary-output-folder>
```
