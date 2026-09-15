# Data provenance and redistribution note

## Bundled daily market snapshots (`data/`)

- `btc.csv` — daily Bitcoin observations, 3 January 2009 to 24 May 2026.
- `eth.csv` — daily Ether observations, 30 July 2015 to 24 May 2026.
- `sp500-2000.csv` — daily S&P 500 prices, 3 January 2000 to 17 April 2020.
- `vix_daily.csv` — daily VIX series used to construct log VIX and the variance-risk-premium features.

The matched real-data attribution uses the S&P feature archive (`tables/features_SPX.csv`, complete 11-feature sample from 30 December 2002); BTC and ETH are used only for archived cross-market dynamical diagnostics.

## Redistribution check before making the repository public

Third-party market-data series are not owned by the author. Before making this repository public:

1. Re-read the licence/terms of use of each data vendor/source and confirm whether redistribution of the raw daily files is permitted.
2. For any file whose redistribution is **not** permitted, replace the raw CSV in the public repository with:
   - a `data/README.md` giving the exact source URL, download steps, date range and checksum of each required series;
   - a small loader/builder script that reconstructs the analysis features and verifies the downloaded files against recorded checksums;
   - keep all frozen **derived** outputs in `tables/` and `tables_csf/` (processed forecasts, metrics and diagnostics, not the raw vendor database), which is what the manuscript tables and figures consume.
3. The journal submission package may still contain the snapshots as a confidential reproducibility archive for reviewers; that is separate from public redistribution.

Do not push data whose licence forbids redistribution to a public repository.
