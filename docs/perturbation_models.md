# Perturbation Models

## Evidence and parameter status

The atmosphere table values below use Vallado, *Fundamentals of Astrodynamics
and Applications*, Table 8-4 (500 km row), as reproduced in
[arXiv:1610.02156](https://arxiv.org/abs/1610.02156). This is a secondary
source; primary verification against the book is pending. Historical numerical
validation and sensitivity outputs later in this document were not regenerated
for the updated atmosphere settings; their applicability to the updated
configuration is `PLACEHOLDER`.

## Scope and assumptions

The linearized relative-motion model was checked against a numerical Jacobian; its physical accuracy beyond the stated scope is not established. The state convention remains `[r_sat, v_sat, r_deb, v_deb]` in SI units,
and relative acceleration is `f_cw + f_perturbation + u`. The perturbation
models assume a circular chief orbit, small deputy separation for the
linearized J2 option, an inertial Earth-fixed rotation rate for drag, and no
third-body gravity, solar radiation pressure, winds, estimation noise, actuator
dynamics, or thrust-execution error.

## J2 model

The full inertial J2 acceleration is the standard zonal expression from
Montenbruck and Gill, *Satellite Orbits*, Springer (2000), Eq. 3.74:

```
a_J2 = (3 J2 mu R^2)/(2 r^5) *
       [ x(5 z^2/r^2 - 1),
         y(5 z^2/r^2 - 1),
         z(5 z^2/r^2 - 3) ]
```

The relative nonlinear acceleration is the deputy value minus the chief value.
The linear model uses the analytic Cartesian Jacobian of this field evaluated at
the chief and transformed into chief LVLH axes. For relative position `rho`,
`a_J2,linear = R_LVLH J_J2 R_LVLH^T rho`. This is the local first-order form
of the Schweighart-Sedwick (2002) linearized J2 relative-motion model, with the
LVLH convention made explicit by the basis rows radial, along-track, and orbit
normal. Higher-order deputy separation terms are omitted.

The analytic Jacobian was checked against central finite differences at five
random chief inclinations and arguments of latitude. The maximum absolute error
was `4.842e-17 m/s^2 per m`; the comparison was scaled by the largest Jacobian
entry and was below `1e-6`.
These numerical checks are historical test outputs; reproduction and artifact
provenance for this documentation update are `PLACEHOLDER`.

## J2 linearized model: validity envelope

The Schweighart-Sedwick linearized J2 relative-motion model was compared against
full nonlinear two-body-plus-J2 integration (DOP853, `rtol=1e-12`) over one
orbit.

| Deputy offset | Max position error | Max velocity error | Within spec (1 m, 1e-3 m/s) |
|---|---:|---:|---|
| 10 m | 0.36 m | 0.022 m/s | Position yes; velocity no |
| 100 m | 3.6 m | 0.22 m/s | No |
| 1000 m | 37 m | 2.2 m/s | No |

The table is a historical recorded test result; current independent
reproduction and artifact provenance are `PLACEHOLDER`.

The linear model is accurate to roughly 10 m offset for position at the chosen
tolerance. Above that, the nonlinear model must be used for J2 effects. The
velocity target is not met even at 10 m, so this is a measured position-only
envelope, not a combined position-and-velocity certification.
This interpretation is limited to the historical test record; its current
reproduction status is `PLACEHOLDER`.

## Atmosphere and storm profile

The local density model is

`rho(h) = rho0 exp(-(h-h0)/H)`.

The function `density_exponential` names its first argument `altitude_m`
([atmosphere.py](../aegis_orbit/envs/perturbations/atmosphere.py), lines 6–25).
The environment forms that argument as
`altitude_above_surface_m(chief_position)` at
[perturbed_conjunction.py](../aegis_orbit/envs/perturbed_conjunction.py), line
167; the helper at line 22 computes centre-radius minus `R_EARTH`. Therefore
`altitude_m` and `h0` are altitude above Earth's surface, not radius from
Earth's centre.

The config uses `rho0 = 6.967e-13 kg/m^3` at `h0 = 500000.0 m` and
`scale_height = 63822.0 m`, all from the cited 500 km table row with status
**SOURCED, secondary; primary verification pending**. The scenario is near
550 km, so its density is an extrapolation of the 500 km reference:
`exp(-(550000 - 500000)/63822)`, about `0.46 * rho0`. This factor is an
extrapolation from the local exponential model, not a separately tabulated
550 km value.

The shape is useful for controlled experiments because it is monotone with
altitude, but it omits composition and solar activity variation. The storm
multiplier is 1 before onset, rises linearly to `peak_factor`, then follows

`1 + (peak_factor-1) exp(-(t-onset-rise)/decay)`.

This represents a stated fast-rise, slow-recovery profile and is not measured
storm data. The storm shape parameters `peak_factor`, `rise_s`, and `decay_s`
remain `PLACEHOLDER`.

## Differential drag

For each object, drag is

`a_drag = -0.5 rho (Cd A/m) |v_atm| v_atm`,

where `v_atm = v_inertial - omega_E cross r`. Differential drag is deputy drag
minus chief drag, transformed to chief LVLH coordinates. Its magnitude depends
on the explicitly supplied coefficient difference; no object coefficients are
sourced here, so the magnitude is `PLACEHOLDER`. The implemented expression is in
[drag_relative.py](../aegis_orbit/envs/perturbations/drag_relative.py).

## Parameters

| Parameter | Value | Unit | Source or status |
| --- | ---: | --- | --- |
| `MU_EARTH` | `3.986004418e14` | m^3/s^2 | WGS-84 Earth gravitational parameter; existing project constant |
| `R_EARTH` | `6378137.0` | m | WGS-84 equatorial radius; existing project constant |
| `J2` | `1.08263e-3` | dimensionless | Standard Earth J2 value; verify exact reference before operational use |
| `omega_earth` | `7.2921159e-5` | rad/s | IERS conventional Earth rotation rate; epoch and polar motion omitted |
| `rho0` | `6.967e-13` | kg/m^3 | Vallado, *Fundamentals of Astrodynamics and Applications*, Table 8-4 (500 km row), reproduced in arXiv:1610.02156; secondary source, primary verification pending |
| `h0` | `500000.0` | m altitude above surface | Same sourced table row; secondary source, primary verification pending |
| `scale_height` | `63822.0` | m | Same sourced table row; secondary source, primary verification pending |
| Chief `Cd*A/m` | `null` | m^2/kg | PLACEHOLDER: requires chief mass, cross-sectional area, and drag coefficient, each with a source |
| Deputy `Cd*A/m` | `null` | m^2/kg | PLACEHOLDER: requires deputy mass, cross-sectional area, and drag coefficient, each with a source |
| storm `peak_factor` | `null` | dimensionless | PLACEHOLDER, storm profile amplitude is not sourced |
| storm `rise_s` | `null` | s | PLACEHOLDER, storm rise duration is not sourced |
| storm `decay_s` | `null` | s | PLACEHOLDER, storm recovery time is not sourced |

The environment raises `ValueError` naming the missing chief or deputy
ballistic-coefficient config key when drag is enabled with a null coefficient.
No coefficient or object property is inferred.

## Perturbed environment and validation

The wrapper keeps the CW/control step exact and integrates only the nonlinear
perturbation correction with ten fixed RK4 substeps per control interval. This
is a splitting method: CW plus constant control is solved analytically, then
the perturbation correction is advanced at fourth order. Its error depends on
step size and perturbation variation; it is not a proof of continuous-time
accuracy. The one-step comparison against a full RK45 integration measured a
maximum state error of `3.977e-8` in the configured test case.

| Test | Check | Tolerance | Measured result |
| --- | --- | --- | --- |
| J2 Jacobian finite difference | Analytic local derivative | relative `1e-6` | max abs `4.842e-17` |
| J2 nonlinear sweep | Full two-body+J2 DOP853 reference | `1 m`, `1e-3 m/s` target | envelope failure documented above |
| Atmosphere endpoints | Reference density and monotonicity | exact/reference ordering | pass |
| Storm profile | Peak, recovery, continuity | peak exact; 1% recovery | pass |
| Drag direction/scaling | Opposing velocity and linear scaling | exact within float64 | pass |
| Disabled wrapper | Matches CW environment | `1e-12 m` | pass |
| RK4 splitting | Full RK45 one-step reference | `1e-6` test threshold | `3.977e-8` |

All measured values in this validation table are historical test outputs;
independent reproduction and artifact provenance for this documentation update
are `PLACEHOLDER`.

The shield and truth models are separate switches. `shield_model="cw_j2"`
selects the nominal `j2_linearized_relative` drift callback, while
`truth_model="cw_j2"` selects the full nonlinear `j2_acceleration_eci`
evaluation for chief and deputy, followed by LVLH differencing. The sensitivity
study therefore used full J2 truth dynamics; the linear validity envelope above
applies only to the nominal linear model. The info dictionary reports both
model values.

## Historical sensitivity outputs

The repository contains a prior sensitivity-run record at
`results/perturbation_sensitivity.csv` and
`results/perturbation_sensitivity.png`. Those outputs used the prior
`PLACEHOLDER` atmosphere settings and are not results for the sourced-secondary
values now in the config. They are not reproduced or interpreted here; their
applicability to the updated settings is `PLACEHOLDER`.

The J2 sweep assertions are regression checks on the recorded envelope, not a
requirement that the linear model remain inaccurate. The 100 m and 1000 m
checks use plus-or-minus 20 percent bands around the measured errors. The
splitting threshold is likewise a documented numerical margin: RK4 with ten
sub-steps over `dt = 1 s` has fifth-order local truncation, and the retained
`1e-6` state-error threshold is approximately 25 times the measured
`3.977e-8` error.

## Limitations

The models omit third-body gravity, solar radiation pressure, thrust-execution
error, higher gravity harmonics, atmospheric winds, density estimation error,
and actuator dynamics. The atmosphere is exponential only. The storm profile
is a stated shape and not measured data. Earth rotation is approximated as a
constant vector. The J2 linear model has the measured validity envelope above,
and the perturbation wrapper's RK4 splitting error depends on the selected
step and parameters. Numerical outputs retained from earlier local tests are
historical records; their applicability to any configuration not stated with
them is `PLACEHOLDER`.
