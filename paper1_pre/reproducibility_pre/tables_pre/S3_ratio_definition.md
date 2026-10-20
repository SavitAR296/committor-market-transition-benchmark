# S3 -- definition of the flux/drift ratio

The archived diagnostic (tables/E2v2_diag_*.csv, column flux_over_drift_median) was produced by code that is not in
the reproducibility package; its state (n_train starts after the 750-day Hawkes window), its drift model (columns
hess_wmin/hess_wmax, n_post, n_subnet indicate an ensemble of non-linear drift fits) and its median convention cannot
be recovered.  We therefore define the ratio explicitly (Appendix D) and recompute it on the common-7 PC state with
the frozen linear OU fit.

Definition: J(z) = f(z) rho(z) - (1/2) div(D rho)(z);  R = median over z ~ rho of |J(z)| / (|f(z)| rho(z)).

Medians over the archived years:

| market   |   archived_ratio |   ratio_analytic |   ratio_grid |     ep |     d1 |
|:---------|-----------------:|-----------------:|-------------:|-------:|-------:|
| BTC      |           0.6605 |           0.1413 |       0.1694 | 0.0016 | 0.0017 |
| ETH      |           0.7024 |           0.1373 |       0.1698 | 0.0016 | 0.0016 |
| SPX      |           0.9946 |           0.4238 |       0.4061 | 0.0177 | 0.0173 |

Interpretation: the analytic and grid versions agree with each other (same fitted model, different quadrature);
differences from the archived values reflect the unrecoverable state and drift model, not the definition.
