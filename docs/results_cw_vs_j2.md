# CW versus CW + J2 comparison results

## Infeasibility

For PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) and PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius), all 2,000/2,000 shield-on episodes became infeasible: 500/500 in each parameter group for each truth model. Mean infeasible steps per episode ranged from 155.6100 to 264.6200, as detailed in the table below. The shield therefore operated only in its infeasible regime. The start states for PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) and PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius) lie inside the barrier by construction of the parameter grid; see [shield_grid.csv](../results/shield_grid_20261014/shield_grid.csv).

## Collision outcomes

For PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius), shield-on collision counts were lower than the shared shield-off count (2/500 versus 15/500) under both truth models. For PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius), shield-on collision counts were higher (20/500 versus 15/500) under both truth models. Shield-off is counted once per truth model; its copies at both parameter labels represent the same 500 episodes.

## Fuel

The following values are mean fuel per condition from `tables.md`; every collision count and denominator shown here was independently recomputed from `shield_comparison.csv`.

| Truth model | Parameter label | Infeasible shield-on episodes | Mean infeasible steps per shield-on episode | Shield-off collisions | Shield-on collisions | Collisions among infeasible episodes | Collisions among episodes never infeasible | Mean fuel, shield off (m/s) | Mean fuel, shield on (m/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cw | PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) | 500/500 | 155.6100 | 15/500 (0.0300) | 2/500 (0.0040) | 2/500 (0.0040) | 0/0 | 0.0497 | 0.0589 |
| cw | PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius) | 500/500 | 264.6100 | 15/500 (0.0300) | 20/500 (0.0400) | 20/500 (0.0400) | 0/0 | 0.0497 | 0.0670 |
| cw_j2 | PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) | 500/500 | 155.6260 | 15/500 (0.0300) | 2/500 (0.0040) | 2/500 (0.0040) | 0/0 | 0.0497 | 0.0590 |
| cw_j2 | PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius) | 500/500 | 264.6200 | 15/500 (0.0300) | 20/500 (0.0400) | 20/500 (0.0400) | 0/0 | 0.0497 | 0.0670 |

## CW and CW + J2 table comparison

For PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius), the CW and CW + J2 mean infeasible-step values differ by 0.0160 (155.6100 versus 155.6260); for PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius), they differ by 0.0100 (264.6100 versus 264.6200). Mean fuel differs by 0.0001 m/s at PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) (0.0589 versus 0.0590) and 0.0000 m/s at PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius) (0.0670 in both tables). These comparisons apply only to these parameter values and this scenario.

## Drag scope

CW + J2 + drag was blocked and not tested because the chief and deputy ballistic coefficients are unsourced. This comparison includes only PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) and PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius), without drag.

## Reproducibility

`manifest.json` contains no config-hash field. As of 2026-10-15, the SHA-256 of `configs/sac_unshielded.yaml` is `098391214ac211765190e2c84b81f60a51ee5c8d9d5ce4cf3aa5eae5c2ec6426`; this is a dated fact about the file, not a hash recorded for the run. The five selected policy digests, verified against the index before evaluation, are listed below. This record covers PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) and PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius).

- Train seed 0, `sac_primary_seed0_step50000.pt`: `f93ee9e47d9bbbce71e32109d67e93533b91f0da3f624bebdf31df102570f2fa`.
- Train seed 1, `sac_primary_seed1_step20000.pt`: `045da28b81aba44e434f43cb235becc48278dcfcd97610f709456c9eba48ea97`.
- Train seed 2, `sac_primary_seed2_step30000.pt`: `d7121edf816b48e67abd9ca1a3eda854ae4633f9d7d9aeda8e05e997261f6f58`.
- Train seed 3, `sac_primary_seed3_step110000.pt`: `00ad81e08bd0884520e23db2b144dbab49c3580c3c6a1f3fdfad7095669a727c`.
- Train seed 4, `sac_primary_seed4_step190000.pt`: `e9aca3ed331e3c462c9830b0cf041c43e945b7fbc7704e7d6ae0447321622cf1`.

The parameter-grid reference is [shield_grid.csv](../results/shield_grid_20261014/shield_grid.csv); it is not an outcome table for this comparison. Step rows are retained in memory during the run.

## Limitations

This is one training run with five training seeds and one architecture, evaluated on seeds 10000–10099. Atmosphere values are secondary-sourced and primary verification is pending; the storm profile remains a PLACEHOLDER. There was no shield-in-the-loop training and no feasible-regime evidence. These limitations apply to PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius) and PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius).
