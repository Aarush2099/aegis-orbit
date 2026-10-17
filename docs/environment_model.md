# Environment Model

## Model and assumptions

The environment is a linearized relative-motion model about a circular
reference orbit. The 12-state convention is
`[r_sat(3), v_sat(3), r_deb(3), v_deb(3)]`, in meters and meters per second,
with a local RTN frame. Relative position and velocity are satellite minus
debris. The model assumes small relative distance compared with the orbit
radius, no J2 or drag, exact state knowledge, and instantaneous piecewise-
constant acceleration.

For radial `x`, along-track `y`, and cross-track `z`, the equations are

```
x_ddot = 3 n^2 x + 2 n y_dot + u_x
y_ddot = -2 n x_dot + u_y
z_ddot = -n^2 z + u_z
```

For `q = [x, y, z, x_dot, y_dot, z_dot]`, these are `q_dot = A q + B u`, with
the upper-right block of `A` equal to `I`, the lower-left block equal to
`diag(3 n^2, 0, -n^2)`, `A[3, 4] = 2n`, `A[4, 3] = -2n`, and `B` equal to
`[0; I]`. This matches the existing CW transition matrix.

## Closed-form forced response

For constant `u` over an interval `t`, variation of constants gives

```
q(t) = Phi(t) q(0) + integral_0^t Phi(s) B u ds
     = Phi(t) q(0) + Gamma(t) u.
```

Because `B` selects the last three columns of `Phi`, integrate those columns
using `sin(nt)` and `cos(nt)`:

```
Gamma_xx = integral cos(ns) ds = sin(nt)/n
```

The position rows use the position blocks of those columns:

```
Gamma[0, 0] = (1-cos(nt))/n^2
Gamma[0, 1] = 2*(nt-sin(nt))/n^2
Gamma[1, 0] = 2*(sin(nt)-nt)/n^2
Gamma[1, 1] = 4*(1-cos(nt))/n^2 - 3*t^2/2
Gamma[2, 2] = (1-cos(nt))/n^2
```

The velocity rows are:

```
Gamma[3, 0] = sin(nt)/n
Gamma[3, 1] = 2*(1-cos(nt))/n
Gamma[4, 0] = 2*(cos(nt)-1)/n
Gamma[4, 1] = 4*sin(nt)/n - 3*t
Gamma[5, 2] = sin(nt)/n
```

All omitted entries are zero. The implementation computes this explicit
six-by-three matrix and does not duplicate the state-transition matrix.

## Discretization and sampling

Each control step is exact for piecewise-constant acceleration: it evaluates
`Phi(dt) q + Gamma(dt) u`. The only numerical approximation in the reported
minimum distance is sampling 20 equal sub-intervals, including both endpoints.
If relative speed is bounded by `V` over the interval, a closest point is at
most half a sample interval from a sampled point, so the reported minimum can
underestimate the continuous minimum by at most `V*dt/40`.

## Validation

| Test | Check | Tolerance | Measured result |
| --- | --- | --- | --- |
| `test_phi_reduces_to_identity_at_zero` | `Phi(0)=I` | `1e-12` | pass |
| `test_gamma_closed_form_matches_quadrature` | Closed form vs `quad` | relative `1e-10`; absolute `2e-10` roundoff allowance | pass; max absolute discrepancy `1.32e-10` |
| `test_zero_thrust_matches_free_cw_propagation` | Free transition | `1e-9 m` | pass |
| `test_constant_thrust_matches_closed_form_forced_solution` | Forced transition | `1e-9 m` | pass |
| `test_environment_matches_independent_linear_ode_reference` | `solve_ivp` over 60 steps | `1e-6 m`, `1e-9 m/s` | `2.085e-12 m`, `3.345e-15 m/s` |
| `test_substep_minimum_distance_has_documented_bound` | Sampled minimum | `V*dt/40` | pass |

The numerical reference comparison is a check against an independent
integration of the same linear model. It does not validate the physics against
reality.

## Limitations

The environment omits J2, atmospheric drag, estimation noise, actuator
dynamics, and thrust-execution error. Linearized dynamics and the continuous-
time model may differ from a future discrete implementation outside their
assumptions. The sampled minimum is an approximation, not a continuous
collision certificate.