# Task B2: start-state feasibility survey plan

## Scope

Plan a future survey of additional `d_safe` parameter candidates beyond the
existing parameter points: PARAMETER-SWEEP d_safe = 30 m (not a derived safe
radius), PARAMETER-SWEEP d_safe = 40 m (not a derived safe radius),
PARAMETER-SWEEP d_safe = 50 m (not a derived safe radius),
PARAMETER-SWEEP d_safe = 60 m (not a derived safe radius), and
PARAMETER-SWEEP d_safe = 67 m (not a derived safe radius). This plan does not
specify or select any additional candidate value.

## Per-start-state test

Before execution, define and preserve the exact training-start cohort using
the configured start-state generation procedure and its seed inputs. For
every proposed parameter and every individual start state `x0`, compute
`h(x0) = ||r0||^2 - d_safe^2` and record the start-state identifier, initial
relative position, parameter label, computed barrier value, and the Boolean
result of `h(x0) >= 0`.

## Acceptance and reporting rules

- Treat each candidate only as a parameter, not as a derived radius.
- Report feasibility for each start state separately. Do not report a single
  feasibility verdict per `d_safe` in place of the per-state records.
- If a summary is included, derive its numerator and denominator directly
  from the per-start-state records and retain the underlying records.
- Do not choose a parameter from this survey. Any later selection requires a
  separately sourced uncertainty definition, risk-multiplier justification,
  and explicit review.
- Do not run this survey as part of this planning task.

This document is a plan only; no start states were evaluated and no additional
parameter values were selected.
