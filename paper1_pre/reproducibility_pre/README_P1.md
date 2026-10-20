# Paper 1 (PRE) -- reproducibility package

Manuscript: *Observer-induced irreversibility in trailing-window early-warning states*, B. Xing and J. Pan
(`../manuscript/main.tex`, Supplemental Material `../manuscript/supplement.tex`, cover letter `../manuscript/cover_letter.tex`).
The frozen CSF-era package `../reproducibility/` is read-only; everything for Paper 1 lives here.

## Run order (`make -f Makefile_P1 all`)
| step | script | output (tables_pre/ unless stated) | seed | runtime |
|---|---|---|---|---|
| data | `E1b_splice_yahoo.py` | `data/spx_2000_2026.csv`, `E1b_splice_report.csv` | -- | 2 s |
| data | `E1e_analysis_data.py` (reads `data/spx_volume_patches.csv`) | `data/spx_2000_2026_r1.csv`, `E1e_repair_log.csv` | -- | 1 s |
| gate | `E1c_data_sanity.py` (exit 1 stops the build) | `E1c_data_sanity.csv` | -- | 2 s |
| data | `E1d_archive_check.py` | `E1d_archive_check.csv` | -- | 2 s |
| theory | `T2_gaussian_observer.py` | `S1_gaussian_leak.csv`, `S2_gaussian_irrev.csv`, `T2_*.csv` | 20261001 | ~20 min |
| theory | `T2b_ep_factorization.py`, `T2c_observer_curves.py` | `T2b_ep_factorization.csv`, `T2c_observer_curves.csv` | -- | 2 s |
| theory | `T3_oneway_driving.py` | `T3_oneway.csv` | 20261002 | 5 min |
| double well | `S4_doublewell_collapse.py` | `S4_*.csv` (frozen phase diagram re-run) | 20260914/20261003 | ~2 h |
| double well | `S4d_two_channel_leak.py` | `S4d_two_channel_leak.csv`, `S4d_acf.csv` | 20261017 | 5 s |
| double well | `S4e_two_rate_leak.py` | `S4e_two_rate_leak.csv` | [20261020, k] | 80 s |
| double well | `S5_doublewell_twowindow.py` | `S5_*.csv` | 20261004 | ~30 min |
| markets | `E4_surrogates.py`, `E4b_leverage.py` | `E4_*.csv`, `E4b_*.csv` | 20261005 | 4 min |
| markets | `E4c_channel_decomposition.py 200 50` | `E4c_*.csv` | 20261005 | 75 s |
| markets | `E4e_decomposition_extensions.py 100 200 200` | `E4e_*.csv` | [20261018, k, sid] | 2 min |
| markets | `E4f_sv_variants.py 30` (helpers in `E4d_sv_null.py`) | `E4f_*.csv` | [20261019, v, k, rep] | 10 s |
| theory | `T2d_table1_horizon.py` | `T2d_table1.csv` (Table I at h = 10, omega_fit) | -- | 4 s |
| markets | `E4g_bootstrap_sensitivity.py` | `E4g_bootstrap_se.csv`, `E4g_volume_sensitivity.csv`, `E4g_pca_loadings.csv` | [20261028, k, L] | 60 s |
| markets | `E9_market_information.py` | `E9_market_information.csv` (Sec. IV G, Table S11) | 20261029 | 85 s |
| output | `fig_p1.py`, `tab_p1.py`, `P1_numbers.py` | `figures_p1/`, `../manuscript/tables/`, `../manuscript/numbers.tex`, `P1_numbers_sources.csv` | -- | 20 s |
| build | `sync_labels.py`, `check_table_numbers.py`, `P1_check_prose.py` (run by `make -f Makefile_P1 manuscript`) | `../manuscript/{supp,main}_labels.tex`; the check fails the build on any numbering mismatch | -- | 30 s |

## Data provenance
* `../reproducibility/data/sp500-2000.csv`: vega-datasets snapshot of Yahoo ^GSPC, 2000-01-03..2020-04-17 (frozen).
* `data/spx_yahoo_chart_api_2019-01-02_2026-10-16.csv`: Yahoo Finance ^GSPC daily OHLCV, downloaded by the authors in
  October 2026 through the chart API (`query2.finance.yahoo.com/v8/finance/chart/^GSPC`).  Its volume is 0 on
  2023-05-24, as in the earlier extension file `data/sp500_2020-04-01_to_2026-10-16_upload.csv` (identical volumes on
  all 1645 common days): a vendor defect.  `E1e` fills it with 3,739,160,000 shares from Tonghuashun (index SPX;
  3,739.16 x 10^4 lots x 100 shares, as reported by Tonghuashun and recorded by the authors in October 2026), recorded
  in `data/spx_volume_patches.csv` together with its cross-checks: that vendor's OHLC for the day and its volumes on
  2023-05-23 (4,155,320,000) and 2023-05-25 (4,147,760,000) are identical to Yahoo's; `E1e` re-checks them on every run
  and stops on any non-positive volume without a documented patch.  A copy of the download with the 2023-05-24 row
  deleted was deliberately not used: dropping a trading day merges two daily returns and shifts every trailing window.
* `data/archive_vijinho_sp500_1950_2018.csv`: independent Yahoo-derived archive (github.com/vijinho/sp500, archived
  2025) used by `E1d` to check the frozen snapshot.
* Bitcoin and Ether: Coin Metrics community data in `../reproducibility/data/`.

## Defect found and fixed during the rebuild
A single zero volume makes `log(volume) = -inf`; in `E4_surrogates.py` the volume Fourier transform then becomes NaN, the
phase randomization of volume stops depending on the seed, and all surrogates share one volume path.  The CSF-era
surrogate bands were affected (published S&P band 0.0006/0.0029/0.0071; repaired 0.0004/0.0027/0.0070; conclusion
unchanged).  `E4_surrogates.py` now asserts strictly positive volumes and reads the gated file.

## Sensitivity of the one patched volume
Replacing the 2023-05-24 volume by the mean of 2023-05-23 and 2023-05-25 (4,151,540,000 shares) leaves the window and
leverage channels unchanged (they do not use volume) and changes the PCA(2) antisymmetries by at most 4e-6 and its
entropy production by 1.4e-5 (`tables_pre/E4g_volume_sensitivity.csv`).

## Why the repository is called *committor-market-transition-benchmark*
The repository began with a benchmark of committor (hitting-probability) forecasts of market stress.  Paper 1 contains
no committor construction: it analyses the trailing-window state on which such forecasts are built.  The committor
forecasts are evaluated in the companion paper (`../paper2_ijf/`).

## Quick rebuild
`make -f Makefile_P1 quick` rebuilds all tables, text macros and the three PDFs from the shipped CSV files (no
simulation, about one minute) and runs the cross-reference check.
