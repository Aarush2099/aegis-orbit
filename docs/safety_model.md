# Safety Model

## Problem setup

The state uses a local RTN-style frame and is represented as
`[r_sat(3), v_sat(3), r_deb(3), v_deb(3)]`, with meters and meters per second.
The relative position and velocity are satellite minus debris. Relative motion
is modeled with Clohessy-Wiltshire dynamics for `[x, y, z, vx, vy, vz]`, where
`x` is radial, `y` is in-track, and `z` is cross-track.

## Barrier derivation

Let `r` be relative position, `v` relative velocity, and `f` the control-free
relative acceleration. The shield uses the parameter `d_safe`, but its
previously configured value was not derived. Let `R_hbr` be the configured
hard-body radius and define the conditional radius

`d_safe = R_hbr + k_sigma * sigma_miss`.

Here `sigma_miss` is the one-sigma uncertainty of relative position at closest
approach. The environment has no covariance model, and this repository has no
cited source for `sigma_miss`. The configuration therefore marks
`sigma_miss_m` as `null` with `PLACEHOLDER`. `k_sigma` is also marked
`PLACEHOLDER` because no project risk-acceptance source specifies it. The
existing numeric `d_safe_legacy_unused_m` config entry is retained only as a
legacy environment-constructor value; it is not used by the shield, is not
computed from this formula, and is not validated. The
uncertainty-derived radius is therefore unavailable until sourced uncertainty
and risk inputs are provided.

The barrier condition at every training start must be

`h(x0) = ||r0||^2 - d_safe^2 >= 0`.

Training starts have radial offsets in `[67.0, 67.5] m`, so the required
derived upper bound is `d_safe <= 67.0 m`. This is a constraint on any later derived value, not a
selected radius. Because `sigma_miss_m` is currently `null`, the start-state
check cannot yet be evaluated.

For a numeric `d_safe`, the barrier is

`h = ||r||^2 - d_safe^2`, with the barrier-admissible set given by `h >= 0`.

## Parameter sweep (Task B)

The safe-radius derivation remains blocked because `sigma_miss_m` and
`k_sigma` are still `PLACEHOLDER` inputs. The following are only the requested
parameter-sweep points; no `d_safe` value is derived, selected, validated, or
claimed safe. The configured training band is read as `[67.0, 67.5] m`.
For each point, the start vector uses the radial-offset convention
`r0 = [offset, 0, 0]`; the sweep samples 200 evenly spaced interior offsets
plus both configured endpoints. `PASS` means only that sampled initial barrier
values are nonnegative.

| Label | Parameter d_safe (m) | Minimum h(x0) (m^2) | Fraction with h >= 0 | Sampled starts | Initial-state check |
|---|---:|---:|---:|---:|---|
| PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS | 30 | 3589 | 1.0 | 202 | PASS |
| PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS | 40 | 2889 | 1.0 | 202 | PASS |
| PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS | 50 | 1989 | 1.0 | 202 | PASS |
| PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS | 60 | 889 | 1.0 | 202 | PASS |
| PARAMETER-SWEEP, NOT A DERIVED SAFE RADIUS | 67 | 0 | 1.0 | 202 | PASS |

These checks cover only the deterministic initial radial vectors and make no
claim about later trajectory behavior.

Then `h_dot = 2 r . v`. With `psi = h_dot + alpha1 h`, the second-order
invariance condition is `psi_dot + alpha2 psi >= 0`. Since `r_dot = v` and
`v_dot = f + u`,

`psi_dot = 2 ||v||^2 + 2 r . (f + u) + alpha1 h_dot`.

Collecting the control terms gives the affine constraint

`a . u <= b`, where `a = -2r` and
`b = 2|v|^2 + 2r.f + alpha1*h_dot + alpha2*psi`.

## QP and exact solution

The shield solves `min_u 0.5 ||u-u_nom||^2` subject to the affine constraint
and the box `|u_i| <= u_max`. For a feasible halfspace, the KKT stationarity
condition gives `u(lambda) = clip(u_nom - lambda*a, -u_max, u_max)` for
`lambda >= 0`. The dot product `a.u(lambda)` is monotone non-increasing, so a
scalar bisection finds the unique boundary multiplier when the nominal action
violates the constraint. This is the exact Euclidean projection because the
objective is strictly convex and the feasible set is closed and convex.

## Assumptions

The model assumes exact state knowledge, no sensor noise yet, accurate drift
modeling, instantaneous thrust, and continuous-time reasoning applied at a 1 Hz
discrete control rate. These assumptions are deliberately explicit rather than
claims about an operational system.

## Feasibility

When the QP is infeasible, the thrust box cannot satisfy the barrier from the
current state. The supervisor reports this condition and applies a heuristic
backup controller. Recursive feasibility is an open question for Phase 2.

## Limitations

There is no proof of safety under sensor noise, unmodeled perturbations, or
discrete-time error. The parameters `u_max`, `alpha1`, and `alpha2` are
placeholders and are not yet justified by a cited thruster model. `d_safe`
remains conditional on the `PLACEHOLDER` uncertainty and sigma multiplier
inputs described above.

## Known closed-loop failure

The 600-step smoke scenario uses `dt = 1 s`, an initial radial separation of
`200 m`, closing velocity `-0.5 m/s`, and a nominal acceleration of
`-u_max * sign(r)` toward the debris. It is a documented failure case, not a
safety proof. With the direct nominal acceleration applied in the historical
smoke integration, the minimum sampled distance was `0.0695 m`.

Running the same scenario through `SafetySupervisor` changed the trajectory:
the minimum sampled distance was `0.1805 m`, with `180` nominal steps, `0`
shielded steps, and `420` backup steps. The final sampled distance was
`104.095 m`. The backup controller therefore changes the outcome and reports
infeasibility, but it does not establish a safe separation or carry a proof of
safety. This is an expected known failure of the current Phase 1 parameters
and discrete-time fallback design.