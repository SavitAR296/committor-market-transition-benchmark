# E1 splice report

* Frozen snapshot: sp500-2000.csv (2000-01-03 .. 2020-04-17, 5105 rows), sha256 9409e9342d0657c747324e4cfabce8a8c7f663bc485b95a3378f36b0a160f8c8
* Extension: author-supplied ^GSPC daily OHLCV, uploaded 2026-10-17 (2020-04-01 .. 2026-10-16, 1645 rows), sha256 6b84f34f2528b386596b20de6a62b0acc3ded62a9aee48d28417dd05313ca264
* Spliced file data/spx_2000_2026.csv: 6738 rows, 2000-01-03 .. 2026-10-16, sha256 fc27dcbb81e5b8cc765fd6cfb4136d5e93cd34e827c15e86ac779c87e39112f0
* Overlap available for comparison: 12 trading days (2020-04-01 .. 2020-04-17); the outline asked for 2019-01-02..2020-04-17, but the supplied extension starts on 2020-04-01.
* Overlap close: max |diff| = 1.170e-04 index points (max relative 4.24e-08); open/high/low max |diff| = 1.170e-04.
* Overlap volume ratio new/old: median 1.0023, range [1.0009, 1.0094].
* VIX: frozen file to 2026-09-04, datahub retrieval to 2026-09-22; max |diff| on 9266 common days = 0.000e+00; values after 2026-09-22 are forward-filled by the frozen loader (full-11 features only).
* Feature equality on 2000-01-03 .. 2020-04-17 (all 28 columns): max |diff| = 3.553e-15; NaN patterns equal: True.
