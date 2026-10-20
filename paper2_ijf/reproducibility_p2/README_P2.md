# Paper 2 (International Journal of Forecasting) -- reproducibility package

Working title: *What does a committor score forecast? Hitting-probability warnings of market stress, their target and
their evaluation*.  The authors.  Companion of Paper 1 (*Observer-induced irreversibility in trailing-window
early-warning states*, PRE), whose package must sit at `../../paper1_pre` (latest version, SHA-checked; never modified).

## Run order (`make -f Makefile_P2 all`)
| script | content | main outputs (tables_p2/) | seed | runtime |
|---|---|---|---|---|
| P2_D0_data_gate | SHA against Paper 1 manifests; patch check of the 2023-05-24 volume; hard failures and warnings; cross-check of the uploaded copy without 2023-05-24 | P2_D0_*.csv | -- | 3 s |
| P2_D1_features | frozen feature code on the gated file; equality with the frozen table 2000-2020; effect of the volume patch | features_SPX_r1.csv, P2_D1_*.csv | -- | 5 s |
| P2_make_e3_frozen | verbatim copy of four E3 helpers into src/p2_e3_frozen.py | -- | -- | 1 s |
| P2_F1_forecasts (--check / default / --window 5) | frozen E3 protocol, 2008-2026; reproduction of the old E3 forecasts; rolling five-year variant; anatomy fields | P2_F1_*.csv | sklearn/deeptime 0 | 45 s / 4.5 min / 2.5 min |
| P2_F2_decomposition --tag exp/roll5 | CORP (MCB, DSC, UNC) and binned Murphy; paired block bootstrap; reliability; annual AUC | P2_F2_*.csv | 20261021 | 20 s |
| P2_F3_drift --tag exp/roll5 | decile drift; saturated-day rates; cluster permutation and quasi-binomial homogeneity tests (day-level chi-square kept once, flagged invalid); real-time calibrators; drift penalty | P2_F3_*.csv | 20261022 | 3 s |
| P2_F4_tests | dBSS with block-bootstrap intervals; NW, EWC fixed-b and Giacomini-White tests; Holm; model confidence set | P2_F4_*.csv | 20261023 | 20 s |
| P2_F5_cases | 2011, 2020, 2025 | P2_F5_*.csv | -- | 2 s |
| P2_F6_anatomy --tag exp/roll5 | S versus the stress coordinate; own target versus operational target; saturation | P2_F6_*.csv | 20261024 | 10 s |
| P2_X1_extension (default / --evaluate) | S&P 500 1957-1999, Bitcoin 2018-2026, Ether 2021-2026 | features_X1_*.csv, P2_X1_*.csv | 0 / 20261025 | 3 min / 45 s |
| P2_figures | draft Figures 1-6 | figures_p2/ | -- | 10 s |
| P2_F1_forecasts --labels-defined, P2_F2 --tag exp_ld, P2_F7_label_rule | sensitivity: drop the 500 rows whose label rule is undefined | P2_F7_label_rule.csv | 20261026 | 5 min |
| P2_B2_baselines | stress index s recalibrated by the same seq-cal map (Tables 1, 4); Giacomini-White split into constant and lagged term; Giacomini-Rossi fluctuation test | P2_B2_stress_index.csv, P2_B5_gw_split.csv, P2_B5_fluctuation.csv | 20261028 | 7 s |
| P2_S1_size_simulation | size and power of the day-level chi-square, episode permutation and quasi-binomial tests in samples that copy the headline sample (Section 5.1) | P2_S1_size_simulation.csv | 20261027 | 75 s |
| P2_numbers | every number quoted in the text as a LaTeX macro, with source table and selection | ../manuscript_p2/numbers.tex, numbers_sources.csv | -- | 2 s |
| P2_tables | Tables 1-6, A1, A2 of the manuscript | ../manuscript_p2/tables/ | -- | 2 s |

Units are cached in tables_p2/_cache (not shipped); `P2_BUDGET` (seconds) limits the units started per call.

## Data
* S&P 500: Paper 1's gated file `spx_2000_2026_r1.csv` (vega-datasets Yahoo snapshot to 2020-04-17, Yahoo chart-API
  download to 2026-10-16).  Yahoo's volume 0 on 2023-05-24 is filled by Paper 1 (E1e) with 3,739,160,000 shares from a
  second vendor whose OHLC and adjacent-day volumes coincide with Yahoo's; P2_D0 re-checks that r1 differs from the
  splice only on that documented day.
* `data/spx_yahoo_2019_2026_without_2023-05-24_upload.csv`: the Yahoo download with the 2023-05-24 row deleted.
  Not an analysis input (Paper 1: dropping a day merges two returns and shifts every trailing window); P2_D0 only
  verifies that it equals Paper 1's Yahoo file on every other day.
* VIX (datahub), Bitcoin and Ether (Coin Metrics), and the 1950-1999 slice of an independent Yahoo-derived archive
  (github.com/vijinho/sp500), all from the Paper 1 package.

## Protocol notes
* Headline: the fixed protocol of E3 (common-7 PCA(2) state, learned target = upper training decile of the stress
  coordinate, OU finite-horizon hitting probability S, strict sequential logistic recalibration), annual expanding
  windows 2008-2026.  Rolling five-year windows and five-year recalibration memory give the finite-window setting of
  the Giacomini-White test.
* AUC is reported only for raw scores and classifiers, never for sequentially recalibrated scores or climatology.
* Label rule: the lagged 1000-day RV quantile needs 500 rows, so in the headline sample labels are structurally
  zero from 2002-12-30 to 2004-12-22 (500 rows); the frozen headline keeps these rows, the extension drops them.
* Extension (fixed before results): first out-of-sample year = first calendar year with at least 1,000 training rows;
  GBT-CV and the logistic C sensitivity are omitted.

## Manuscript
`../manuscript_p2/main.tex` (first draft, article class with natbib author-year; build with `make -f Makefile_P2 manuscript`).
The text contains no hand-typed result number: each one is a macro written by P2_numbers.py, listed with its source
table in numbers_sources.csv.  F7 (label-rule sensitivity) changes no conclusion: the recalibrated committor forecast is
unchanged and no comparison becomes significant after Holm adjustment.

## Revision of October 2026 (outline sections 7 A1-A11, B1-B10)
* Every moving-block bootstrap uses blocks of 2h trading days (`p2_common.boot_block`; 10 days at h = 5, 40 at h = 20);
  F2 and F7 use 2,000 resamples.  `P2_F4_dbss.csv` keeps blocks of 5-60 days for the sensitivity table (Table A3).
* Table A1's headline column takes the bootstrap quantile of Section 4 from `P2_F2_bootstrap_exp.csv`.
* `src/episode_tests.py`: the episode-level tests of Section 5 as a reusable function.
* Runtimes on one core: F2 95 s, F6 25 s, F4 25 s, F7 70 s, X1 --evaluate 95 s (run them in the foreground; detached
  background jobs are not supported in every sandbox).
