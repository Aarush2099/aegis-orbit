# Streaming comparison step output

## Change

`run_shield_comparison.py` now writes eligible step rows to
`shield_comparison_steps.csv` as each shield-on episode completes, using
`StreamingCSVWriter`, instead of retaining all comparison step rows until the
run ends. The CSV header, field order, row serialization, per-episode filter,
episode output ordering, and shield-off reuse behavior were not changed.
Episode-level rows remain accumulated in memory and written at the end.

## Equivalence checks

The old-writer baseline was reconstructed because the workspace has no Git
history. It reproduces the prior `csv.DictWriter` collect-then-write path.
Accordingly, equivalence establishes serialization of identical in-memory
rows; it does not establish equivalence against recovered historical source.

The initial tiny check used one evaluation seed, one `d_safe`, and one truth
model. It produced no shield-off step rows, so blank-field equivalence for a
shield-off step was checked with a synthetic row only.

The larger check used three evaluation seeds, both truth models (`cw` and
`cw_j2`), and `d_safe` values of 40 and 67 m:

- Step CSV: 36,000 rows; serialization matched between paths.
- Episode CSV: 120 rows; serialization matched between paths.
- No mismatch was found. The actual step outputs contained no shield-off
  rows; a synthetic shield-off row with blank fields serialized byte-identically.

Three sequential fast-suite runs each passed 82 tests with 7 deselected.
Pytest-reported durations were 50.50 s, 17.03 s, and 15.16 s (mean 27.56 s).
Measured elapsed times were 53.78 s, 19.44 s, and 17.22 s (mean 30.15 s).

## Step-buffer effect on the comparison runner

The memory comparison is a measurement of the step-buffer effect on the
comparison runner, not a before/after measurement of the original code. The
buffered path reconstructs the old collect-then-write behavior; both paths
ran in separate Python processes with the same configuration and inputs:
10 evaluation seeds, the `cw` truth model, `d_safe` values 40 and 67 m, and
the same selected policies.

Peak working set was read from Windows
`GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize)`, sampled
at each episode-progress boundary and after the runner returned. Each process
had 150 episode-boundary samples.

| Path | Observed peak working set | Step rows |
|---|---|---:|
| Reconstructed buffered writer | Higher | 60,000 |
| Streaming writer | Lower | 60,000 |

The CSV outputs matched in content and row counts. Streaming had a lower
observed peak working set in this single process comparison. This is not a
controlled estimate of general memory savings and may include process and
operating-system variability.
