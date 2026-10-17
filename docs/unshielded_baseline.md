# Unshielded Baseline

Step 2C evaluates an unshielded SAC controller against passive, random, and
finite-oracle references. The CBF shield is deliberately absent; shield-on
comparison belongs to Step 2D.

Earlier perturbation sensitivity artifacts were moved to
`results/superseded/20261011_143208/` before this run. They are not Step 2C
results and are retained only for provenance.

## Distribution

Training samples `initial_offset_m` uniformly from `[67.0, 67.5] m` using the
environment's explicit `numpy.random.Generator`. This band was confirmed on 100
encounters: at 67.0 m passive collision rate was 69% and the worst oracle margin
above the 10 m hard-body radius was 0.426 m. The 67.5 m endpoint had 36% passive
collision rate and 0.589 m worst oracle margin. The model uses `dt=1 s`, 600
steps, closing speed `0.5 m/s`, reporting radius `106.7 m`, and hard-body radius
`10 m`.

## Reward

The training reward is

`-1000 ||u|| - 100 max(0, d_safe-distance) - 100 I(collision) + 1 I(horizon survived)`.

The fuel term is a proxy in m/s^2 per step, the proximity term shapes learning,
and the collision penalty marks the hard-body event. Metrics reported by the
benchmark are unshaped collision rate, distance, fuel, and clipping rate. The
proximity term must be ablated in a later report.

## Oracle and baselines

The oracle evaluates 29 constant Cartesian controls at `-u_max`, `0`, and
`+u_max`, plus two radial sign controls. It is a finite lower-bound feasibility
reference and maximizes miss distance without considering fuel. In the confirmed
67.0 m distribution it avoided all 69 passive collisions, with worst margin
0.426 m. This is a constant-thrust ceiling, not a ceiling for general control.
Passive uses zero thrust. Random uses an explicit seeded Generator and uniform
per-axis actions.

The run's `oracle_avoidable_subset` is defined by passive collisions in its
ordinary band-sampled baseline episodes. For the verified run, that subset has
53 passive-colliding encounters, and the finite oracle found clearance for all
53. This is separate from the 69 passive collisions reported by
`oracle_preflight`: both use evaluation seed identifiers 10000 through 10099
and the strict collision test `min_distance_in_step < hard_body_radius_m`, but
they do not use the same initial offsets. The pre-flight fixes the radial
offset at 67.0 m; ordinary baseline episodes sample the configured
67.0–67.5 m band. Their reset random streams also differ because the
band-sampled environment draws the offset before drawing lateral jitter,
whereas the fixed-offset pre-flight does not draw an offset. Thus the
53-encounter oracle subset is the passive-colliding subset of the band run,
not a re-count of the fixed-offset pre-flight.

Code locations: `benchmarks/run_unshielded_benchmark.py` lines 59–72 create
the band environment, lines 94–114 implement the fixed-offset pre-flight
(including configured seed range and strict collision threshold), and lines
678–697 build the run's passive-collision subset. The shared strict
`min_distance_in_step < hard_body_radius_m` collision rule is at lines
113 and 195–197. `aegis_orbit/envs/continuous_conjunction.py` lines 63–87
show the offset and lateral-jitter draws during reset.
`benchmarks/oracle_reference.py` lines 100–112 filter oracle summaries to
passive-colliding encounters.

## Results

The benchmark writes `results/unshielded_benchmark.csv` with seed and config
hash. It reports mean collision rate, minimum distance, fuel, clipping fraction,
and bootstrap 95% intervals. SAC runs use training seeds 0 through 4 and are
compared on evaluation seeds 10000 through 10099. The benchmark must report the
67.5 m endpoint separately from the uniform-band result.

## Training/evaluation gap

Training and evaluation do not use the same action distribution. During
warmup, training draws uniform random actions; afterward it calls
`SACAgent.action` with its default stochastic mode, which samples from the
actor. Evaluation calls `deterministic_evaluation_policy`, which selects the
actor mean after the `tanh` transform. This is intentional for repeatable
policy evaluation, but training-episode collision frequency is not a
deterministic-policy evaluation metric.

The existing `results/training_log.csv` contains 2,321 `sac_primary` training
episodes, 836 marked collided (36.02%). The existing
`results/unshielded_benchmark.csv` contains 100 deterministic evaluation
episodes per training seed: seed 0 had 0 collisions, seed 1 had 15, and seeds
2-4 had 0. These counts describe the files already present; they are not a new
benchmark run or a claim that the Step 2C gates passed.

The action-selection paths are in `benchmarks/run_unshielded_benchmark.py`
(`train`, `deterministic_evaluation_policy`, and its evaluation call) and
`aegis_orbit/agents/sac_agent.py` (`SACAgent.action`).

## Interrupted run

The interrupted run directory `results/rerun_primary_20261013/` is preserved
as evidence and its partial files are not used by the restart or policy
verification. Its last progress line was
`[121.9s] training start method=sac_primary train_seed=0`; the log file's last
write time was 2026-10-13 20:37:53.937 +05:30. The process was reported absent
at 01:35 AM, and a read-only process check before restart also found no matching
runner process.

The System event-log query from 2026-10-13 20:00 onward returned unrelated
network/service events, but no shutdown, restart, or power events. The detached
restart was launched to `results/rerun_primary_20261014/` at
2026-10-14 01:42:51 +05:30. Its first progress line was
`[67.8s] oracle pre-flight passed passive_collision_count=69 oracle_avoidable_count=69 minimum_margin_m=0.426314 preflight_elapsed_s=67.8`.

## Verified retraining run

The verified retraining directory is `results/rerun_primary_20261014/`. The
interrupted directory `results/rerun_primary_20261013/` was not used because it
contains only partial output from an interrupted run.

The config hash is `75f49d8ad691`. Verification reported 19 indexed files
hashed, 0 hash mismatches, 5 final policies re-evaluated, 500 final-policy
episodes, and maximum difference 0. Intermediate selected checkpoints are
hashed but are not re-evaluated against the benchmark, per the option 1
verification contract.

The reproduced primary result is a 0.03 collision rate with CI 0.016-0.046,
matching the earlier run on the same evaluation seeds. The benchmark file in
this run predates the oracle-summary CSV format fix.

## Limitations

This is a linearized CW environment with no perturbations, sensor noise,
actuator dynamics, or execution error. The training budget is short, only one
architecture is used, and no broad hyperparameter sweep is performed. Oracle
labels are finite-search lower bounds. Results are collision metrics for the
specified hard-body radius, not guarantees about other geometries.
