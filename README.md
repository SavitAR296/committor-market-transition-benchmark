# Committor market-transition benchmark

Reproducibility repository for two companion manuscripts by Boyan Xing and Jiayi Pan.

| manuscript | journal | directory |
|---|---|---|
| *Observer-induced irreversibility in trailing-window early-warning states* | prepared for submission to *Physical Review E* | [`paper1_pre/`](paper1_pre/) |
| *What does a committor score forecast? Hitting-probability warnings of market stress, their target and their evaluation* | prepared for double-blind submission to the *International Journal of Forecasting* | private until acceptance (see below) |

Paper 1 analyses what trailing windows of prices and volumes do to the irreversibility and
information content of an early-warning state. Paper 2 evaluates a committor-type stress score built
on that state: its target, its calibration, and inference at the level of episodes. The two papers
share the data pipeline; Paper 1's gated data files are inputs to Paper 2.

## Status of the archived draft (important)

An earlier single-author draft, *Committor benchmarks for market regime transitions: state
information, rolling-window failure, and out-of-sample limits*, was **never submitted to any
journal**. The v1.0.0 Zenodo record (`10.5281/zenodo.22774226`, concept DOI
`10.5281/zenodo.22774225`) nevertheless carried the statement "submitted for publication (Chaos,
Solitons and Fractals)", which is **erroneous**. The Zenodo record has been corrected with a note to
that effect and superseded by version 2.0.0 (version DOI `10.5281/zenodo.23491759`, published
2026-10-20; concept DOI `10.5281/zenodo.22774225`, see [`ZENODO_DEPOSIT.md`](ZENODO_DEPOSIT.md)).

The draft's frozen code, data and results are preserved byte-identical (SHA-256-manifested) at
`paper1_pre/reproducibility/` (read-only), and its top-level documentation is preserved in
[`archive_csf_draft/`](archive_csf_draft/). The draft is kept only as a historical record; the two
papers above supersede it.

## Repository layout

```
archive_csf_draft/            # never-submitted single-author draft: documentation + pointers
paper1_pre/
  manuscript/                 # Paper 1 LaTeX sources, tables, figures, compiled PDFs
  reproducibility_pre/        # Paper 1 scripts, frozen tables, figures, data, SHA-256 manifest
  reproducibility/            # frozen legacy package (read-only inputs of Paper 1; see README_P1.md)
run_all.sh                    # two-tier rebuild for both papers
ZENODO_DEPOSIT.md             # Zenodo v2.0.0 deposit + old-record correction steps
CITATION.cff                  # citation metadata (v2.0.0)
LICENSE                       # MIT, both authors
```

## Reproducing the results

Python 3.12 (Paper 1) is tested; see each package's `requirements.txt`.

```bash
# Tier 1 (minutes): rebuild every table, text macro and figure from the shipped frozen CSVs,
# then recompile the manuscripts.
./run_all.sh quick

# Tier 2 (hours, one core): recompute every simulation and fit from the raw data.
./run_all.sh full

# Verify the shipped tree against the SHA-256 manifests.
./run_all.sh check
```

`run_all.sh` dispatches to `make -f Makefile_P1` (Paper 1) and, once public, `make -f Makefile_P2`
(Paper 2). The compiled PDFs in `paper1_pre/manuscript/` correspond to the shipped sources; rebuild
them with `make -f Makefile_P1 manuscript` if a TeX distribution is available.

## Data and redistribution

Third-party market data are **not** redistributed as raw vendor files in this repository:

- S&P 500: the frozen vega-datasets snapshot of Yahoo Finance `^GSPC` (2000-01-03 to 2020-04-17) is
  bundled under its BSD-style vega-datasets license (`paper1_pre/reproducibility/data/`); the
  2020-04-01 to 2026-10-16 continuation was downloaded by the authors from Yahoo Finance's chart
  interface in October 2026. A small frozen snapshot of exactly that window (the raw chart-API file
  used to build the paper's input, ~6.5 years of daily OHLCV) is bundled as a reproducibility
  convenience; a download script (`E1b_splice_yahoo.py`), the data gate (`E1c_data_sanity.py`), the
  repair log for the 2023-05-24 zero-volume day (`E1e_repair_log.csv`, `E1b_splice_report.csv`) and
  SHA-256 checksums are provided so the file can be regenerated from the source. No vendor database
  beyond this frozen window is redistributed.
- VIX: datahub finance-vix series, retrieved 2026-10-17 (frozen copy bundled, source stated).
- Bitcoin and Ether: Coin Metrics community network data (frozen copies bundled, source stated).
- S&P 500 1950-2018: independent Yahoo-derived archive (github.com/vijinho/sp500), archived 2025.

See `paper1_pre/reproducibility_pre/README_P1.md` (Data provenance) for the full record, and
`archive_csf_draft/DATA_REDISTRIBUTION_NOTE_legacy.md` for the original policy note.

## AI assistance disclosure

- Paper 1 and Paper 2 (current manuscripts): Claude (Anthropic) was used to edit the text and AI
  coding assistants to develop analysis code; the authors ran every script and take full
  responsibility. See the manuscripts' declarations.
- The legacy single-author draft (archived): OpenAI ChatGPT (GPT-5.6 Sol) assisted with drafting and
  refactoring code for the archived experiments; see `archive_csf_draft/`.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the manuscripts are accepted, cite the repository itself;
after acceptance, update `CITATION.cff` and the manuscript data-availability paragraphs with the
journal DOIs. The arXiv identifiers of the two papers, once posted, replace the "submitted to ..."
cross-references between them.
