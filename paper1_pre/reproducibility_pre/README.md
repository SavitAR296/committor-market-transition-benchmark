# Reproducibility package v2 — PRE version of Paper B

Manuscript: *Trailing-window observation of transition dynamics: exact information loss and observer-induced
irreversibility, with an application to market stress* (`../manuscript/main.pdf`, Supplemental Material
`../manuscript/supplement.pdf`).  The CSF-era package is kept unchanged in `../reproducibility/` (frozen inputs and
results; documentation fixes only, see `CHANGELOG_legacy.md`); everything new lives here.

## Quick start
```bash
python -m pip install -r requirements.txt     # tested with Python 3.12.3
make check      # sha256sum -c SHA256SUMS.txt
make frozen     # tables + figures from frozen inputs and tables_pre/*.csv  (~2 min)
make all        # recompute every CSV, then `make frozen`                     (~45 min on one core)
make manuscript # latexmk main.tex and supplement.tex in ../manuscript
```
Scripts are resumable: units are cached in `tables_pre/_cache/` (not shipped; delete to recompute) and the environment
variable `PRE_BUDGET` (seconds) only limits how many units start in one call.  Every unit has its own seed stream, so
interrupting and resuming does not change any number.

## Scripts (inputs → outputs; seeds)
| script | content | outputs (tables_pre/) | seed |
|---|---|---|---|
| T1_ledger_check | Murphy ledger on frozen forecasts; all 59 pilot numbers; seq-cal reconstruction; link/eps sweep | T1_*.csv | – |
| T2_gaussian_observer | Prop. 2 closed forms (Table I), covariance check, exact Mehler leak, spectral KLD, S1 leak and S2 irreversibility simulations | T2_*.csv, S1_gaussian_leak.csv, S2_gaussian_irrev.csv | 20261001 |
| T3_oneway_driving | Prop. 3 closed form vs exact-discretization simulation | T3_oneway.csv | 20261002 |
| E1_extend_data | splice frozen S&P snapshot with the 2020-04-01..2026-10-16 file; VIX update; rebuild features with frozen code; equality check | features_SPX_2026.csv, E1_* | – |
| E2_event_log | operational entries and clusters | E2_event_log.csv, E2_clusters.csv | – |
| E3_matched_estimators | annual expanding windows 2008–2026: OU, MSM (deeptime), AMC, logistic, GBT, GBT-CV, C sensitivity; reproduces frozen forecasts to 1e-16 | E3_*.csv | 0 (sklearn) |
| E4_surrogates | decisive irreversibility test: observed, reversed returns, reversible/literal IAAFT, centered windows, Prop.-2 prediction; SPX, BTC, ETH | E4_*.csv | 20261005 |
| E4b_leverage | random signs on actual |r| (leverage removed); leverage function | E4b_*.csv | 20261005 stream 9 |
| S4_doublewell_collapse | frozen phase diagram regenerated (frozen seed sequence, 6 reps) + direct leak, lam_bar, Gaussian limit, fits | S4_*.csv | 20260914 / 20261003 |
| S4b_effective_variance, S4c_scaling_fit | measured posterior variance; empirical two-parameter scaling | S4b_*, S4c_* | 20261003 stream 9 |
| S5_doublewell_twowindow | currents seen by a two-window observer; latent and Prop.-2/3 comparisons | S5_*.csv | 20261004 |
| S3_ratio_definition | explicit flux/drift ratio (analytic and grid) on market states | S3_* | – |
| E5_murphy_inference | Murphy (fixed and decile bins), reliability, annual AUC, block bootstrap, Giacomini–White | E5_*.csv | 20260915 |
| E6_score_drift, E7_cases, E8_pca_geometry | score drift (bootstrap seed 20260916), case studies, PCA geometry | E6_*, E7_*, E8_* | 20260916 / – |
| fig1_pipeline … fig6_cases, make_figures_pre | Figures 1–6 and C1 (vector PDF + PNG) → figures_pre/ | – | – |
| tab1_gaussian, tab2_matched, tab3_irreversibility, make_tables_pre | Tables I–III and S1–S12 → ../manuscript/tables/ (+ tab2/tab3 CSVs) | – | – |
| run_queue | driver used in a 300-s-per-command sandbox; not needed with `make all` | – | – |

## Data provenance
* `../reproducibility/data/sp500-2000.csv`: vega-datasets snapshot of Yahoo Finance ^GSPC, 2000-01-03..2020-04-17 (frozen).
* `data/sp500_2020-04-01_to_2026-10-16_upload.csv`: ^GSPC daily OHLCV supplied by the author on 2026-10-17
  (**source and download date to be recorded by the author before deposition**); validated on 12 overlapping days.
* `data/spx_2000_2026.csv`: spliced file (frozen snapshot through 2020-04-17, then the supplied file).
* VIX: `../reproducibility/data/vix_daily.csv` (frozen, to 2026-09-04) and
  `data/vix_daily_datahub_retrieved_2026-10-17.csv` (datahub finance-vix, to 2026-09-22); merged in `data/vix_2000_2026.csv`.
* Bitcoin and Ether: Coin Metrics community data in `../reproducibility/data/` (BTC first price 2010-07-18; used from 2013-01-01; ETH from 2016-03-01).

## Where each manuscript item comes from
Table I `T2_table1_closed_form.csv`; Table II `tab2_matched.csv` (frozen headline sample for the first two blocks);
Table III `tab3_irreversibility.csv`; Fig. 2 `S1_*`, `S2_*`, `E4_*`, `E4b_*`; Fig. 3 frozen `B_synthetic_phase.csv`, `S4_*`,
`S4b_*`, `S4c_*`, `S5_*`; Fig. 4 `E5_*`; Fig. 5 `E6_*`; Fig. 6 `E7_*`; Fig. C1 frozen `B_fig1_doublewell.pdf`.
Before submission: update the Zenodo record and commit hash so that the manuscript URL and this README agree.
