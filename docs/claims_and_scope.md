# Claims and scope standards

1. Jacobian validation: the 5e-17 figure is the numerical error between the analytic and finite-difference Jacobians. It validates the linearized coefficients, not physical position accuracy.
2. Physical fidelity of the linearized J2 model is reported as the measured envelope against full nonlinear two-body-plus-J2 integration, at the tested offsets only: about 0.36 m at 10 m, 3.6 m at 100 m, and 37 m at 1,000 m relative offset, for one orbit with the stated geometry. The envelope is not a bound at other offsets.
3. The comparison is against full nonlinear two-body-plus-J2 integration. It is a more accurate model, not reality. It excludes drag, third-body effects, and any J2 coefficient uncertainty.
4. cw_j2 runs evaluate the shield under intentional model mismatch: the environment uses full nonlinear J2 acceleration, and the shield's drift model is CW-only in all runs; cw_j2 runs therefore evaluate the shield under mismatch between full nonlinear J2 dynamics in the environment and the CW prediction in the shield. Every cw_j2 table header states: "Shield drift model: CW-only. Environment truth: full nonlinear J2 acceleration. The shield operates under model mismatch in this table."
5. Equal-ballistic-coefficient cases are not physical drag results. They may appear only as a code-path identity test confirming identical inputs give zero relative acceleration, and never in a results table.
6. Valid truth_model values are "cw" and "cw_j2" only. "cw_j2_drag" is not a valid identifier; drag is blocked until ballistic coefficients are sourced.
7. The atmosphere values remain marked secondary; primary verification pending.
