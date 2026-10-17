# Post-hoc shield comparison

## Protocol and scope

This comparison is post-hoc on the five final selected `sac_primary` actor
checkpoints from `results/rerun_primary_20261014/policies/index.csv`, chosen by
the largest selected step for each training seed. The runner checks each
checkpoint SHA-256 against the index before loading it. No shield-in-the-loop
training was performed.

The completed scope requested for launch is exact CW only (`truth_model="cw"`),
using evaluation seeds 10000–10099, both `shield_off` (raw deterministic actor)
and `shield_on` (actor through `SafetySupervisor`), and the `d_safe` parameters
40 m and 67 m. The shield drift model is CW. `d_safe` values are parameters,
not derived radii; every episode CSV record carries
`PARAMETER-SWEEP d_safe = X m (not a derived safe radius)`.

Shield-off episodes are computed once per policy and evaluation seed, then
reused for both parameter values. The manifest records that reuse, and paired
condition initial-state vectors are compared by SHA-256.

The CW + J2 + drag condition is blocked, not run, because
`drag ballistic coefficients and atmosphere settings not sourced; PLACEHOLDER`.
No drag settings or perturbation defaults are changed.

## Outputs

- `shield_comparison.csv`: episode rows and per-episode gate evidence.
- `shield_comparison_steps.csv`: per-step modes only for episodes with at least
  one infeasible step.
- `manifest.json`: run scope, selected policy hashes, row count, and the blocked
  drag condition.
- Manifest entry: **CW + J2 + drag: blocked, ballistic coefficients not sourced.**
- `run.log`, `run.log.err`, `pid.txt`, and `launch.json`: detached-launch
  evidence.
- `tables.md`: pending full-run completion and passing gate checks.

The one-off two-seed dry run wrote 40 episode rows (20 per condition), with
matching initial-state hashes. Its artifacts were removed before the full-run
launcher; no dry-run result is used as a comparison result.

## Gates

- C1 policy preflight: **PASS** — 19 index rows; five final primary checkpoints
  selected; all five hashes matched; output directory was absent.
- C2 runner and two-seed dry run: **PASS** — required columns were present,
  40 rows were written, and initial-state hashes matched. The manifest marked
  CW + J2 + drag blocked with the required reason.
- C3 tests: **PASS** — comparison fixtures, mode counts, hash assertion,
  unsupported-parameter rejection, non-empty output refusal, and gate-checker
  fixture tests passed.
- C4 six full-run checks: **PASS** — all six checks passed after the detached
  run completed; see `results/shield_comparison_20261014/run.log` and
  `results/shield_comparison_20261014/tables.md`.
- C5 detached full-run launch: **PASS** — PID `53144` was alive at launch with parent
  PID `55464`; `run.log.err` was empty, the output-directory argument matched,
  and the first progress line was `[6.5s] physical_episode=1/1500 completed`.
  Launch time was `2026-10-14T15:40:04.9110696+05:30`. The first ten physical
  episodes took 19.252 s (1.925 s/episode), giving a linear estimate of
  2887.8 s (about 48.1 minutes) for 1500 physical episodes. This is a runtime
  estimate, not a comparison result.
- C6 tables: **PASS** — tables were written only after all C4 checks passed.
  Shield-off was computed once (500 episodes) and reused for both `d_safe`
  parameters; see `results/shield_comparison_20261014/tables.md`.

This protocol reports run mechanics and gate state only. It makes no conclusion
about whether the shield helps.

## CW versus CW + J2 comparison

The new comparison is post-hoc on the same five final selected `sac_primary`
checkpoints, with SHA-256 checked against the policy index before use. It does
not retrain policies and uses no shield-in-the-loop training. It compares
`truth_model="cw"` with `truth_model="cw_j2"` at evaluation seeds 10000–10099,
under `shield_off` and `shield_on`, with parameter values 40 m and 67 m.

The shield drift model is CW-only in both conditions and both truth models.
The `cw_j2` environment uses full nonlinear J2 acceleration; its table header
must read: "Shield drift model: CW-only. Environment truth: full nonlinear J2
acceleration. The shield operates under model mismatch in this table."
Every table labels each parameter as
`PARAMETER-SWEEP d_safe = X m (not a derived safe radius)`.

Shield-off episodes are physically evaluated once per policy, evaluation seed,
and truth model, then reused at both d_safe values. The expected full output is
2,000 episode rows per truth model and 4,000 total; each condition/parameter/
truth-model group contains 500 rows. The existing ten-substep RK4 integration
is unchanged. For the perturbed environment, `min_distance_in_step` is
measured at its RK4 start and midpoint stages plus the final state, matching 20
equal sub-interval samples without changing state updates.

CW + J2 + drag remains blocked with reason
`drag ballistic coefficients and atmosphere settings not sourced; PLACEHOLDER`.
The output manifest records this blocked condition. The new run's full-run gate
results and tables remain pending completion.
