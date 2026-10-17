# Task 1: request for sourced uncertainty and risk inputs

No value is proposed in this document.

Before deriving a shield parameter from uncertainty, request a sourced
definition of `sigma_miss` with all of the following:

- **Quantity:** one-sigma relative-position uncertainty at closest approach,
  expressed in meters, with the contributing axes specified. State whether it
  is a one-dimensional projection, an axis-wise covariance description, or a
  scalar derived from a documented covariance convention; do not conflate
  these definitions.
- **Acceptable source type:** orbit-determination accuracy, tracking-data
  covariance, or a published CDM uncertainty sample with its provenance.
  Include the source, applicable population or operating context, and the
  method used to obtain the closest-approach uncertainty.
- **Risk multiplier:** request a `k_sigma` choice and its justification from
  the project risk-acceptance owner or a cited risk policy. The choice must be
  tied to the stated uncertainty convention and decision context.
- **Acceptance test:** once sourced inputs are supplied, test
  `h(x0) = ||r0||^2 - d_safe^2 >= 0` for every configured training start in
  the full training band. Report each start-state result and reject the
  derived parameter if any training start fails the inequality.

Until the source, convention, and justification are available, leave the
uncertainty and multiplier inputs as `PLACEHOLDER`; this request does not
derive or select any `d_safe` parameter.
