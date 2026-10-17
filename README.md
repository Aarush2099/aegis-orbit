# AegisOrbit

AegisOrbit is a research prototype for satellite conjunction simulations using
Soft Actor-Critic (SAC), Clohessy-Wiltshire relative dynamics, optional J2/drag
perturbations, and a control-barrier-function (CBF) action filter.

## Research Status

This code is for controlled numerical experiments, not operational flight use.
The CBF filter can project actions only when its barrier halfspace intersects
the configured actuator box. Infeasible steps use a heuristic backup that has
no safety proof. In the simulator results, shield decisions were infeasible
for the original geometry; the shield therefore did not demonstrate safe
operation in that setting. Baseline and shield results are simulator results,
not evidence of real-world performance. Detailed root-cause reports are local
analysis artifacts under `results/` and are not included in a public source
clone.

The physical `sigma_miss` estimate and derived safe radius remain unresolved
placeholders. The official configuration keeps `sigma_miss_m` null, and
exploratory `d_safe` values are not accepted physical safety thresholds. Drag
comparisons requiring unsourced ballistic coefficients are not included.
The configuration under
`configs/experiments/sigma_shield_sensitivity/experiment.json` is an
exploratory simulator sensitivity specification, not an official configuration
or a validated safety setting.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Tests

Run the fast suite:

```powershell
python -m pytest -m "not slow" -q
```

Run tests marked slow separately:

```powershell
python -m pytest -m slow -q
```

`tests/test_launch_script.ps1` is a separate Windows PowerShell test; pytest
does not run it. It checks that the retraining launcher refuses a non-empty
output directory. Run it from the repository root after creating the
installation above:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\test_launch_script.ps1
```

## Example Commands

The perturbation sensitivity command requires an output directory and refuses
to overwrite existing artifacts:

```powershell
python -m benchmarks.perturbation_sensitivity --output-dir "$env:TEMP\aegis-orbit-perturbation"
```

The main benchmark can train/evaluate policies and write results to the
specified output directory. Use a separate temporary configuration/output
directory for diagnostic runs; do not point exploratory runs at historical
result folders.

```powershell
python -m benchmarks.run_unshielded_benchmark --config <temporary-config.yaml> --output-dir <temporary-output-folder>
```

The PowerShell launchers also require an explicit `-OutputDir`. Choose a
dedicated path such as `results\my-run`; these generated outputs are ignored
by Git and are not part of the public release. The shield-comparison launcher
uses `results\rerun_primary_20261014\policies` as its default policy input,
which is not included in a public clone, so provide an available `-PolicyDir`
when using it.

## Reproducibility and Included Data

The public source repository contains 85 files. A clone does **not** contain
the trained actor checkpoints (`*.pt`), historical result tables/logs under
`results/`, or the ESA/Kelvins archive under `data/`. Consequently, historical
numbers cannot be reproduced from the public repository alone: reproducing
them requires the exact selected policy weights and evaluation artifacts.
Dataset investigations additionally require the separately obtained archive.
The source code, configs, test suite, and reproduction scripts are included;
new runs write outputs that should be identified by their manifest,
policy/config hashes, and seed ranges.

## License

This repository is distributed under the MIT License; see [LICENSE](LICENSE).
Before publishing, the copyright holder should confirm that all contributors
have authority to license their contributions under these terms.