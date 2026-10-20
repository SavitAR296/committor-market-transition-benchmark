# Paper 2 — What does a committor score forecast? Hitting-probability warnings of market stress, their target and their evaluation

B. Xing and J. Pan. Prepared for double-blind submission to the *International Journal of
Forecasting*.

**Status: private during review.** Under IJF's double-blind policy the code and data of this paper
are provided as *anonymous supplementary material* at submission; a public archive (directory
`paper2_ijf/` inside the companion repository) is deposited upon acceptance. This working copy
contains identifying information and must not be made public before acceptance.

## Layout

- `manuscript_p2/` — LaTeX sources (anonymized for submission: the author block, CRediT, funding and
  acknowledgment paragraphs are moved to the title page / submission system), compiled PDFs, tables
  and figures.
- `reproducibility_p2/` — scripts, frozen tables, figures, data and SHA-256 manifests. See
  `reproducibility_p2/README_P2.md`.
- The package requires the Paper 1 package `paper1_pre/` as a sibling in the same repository root
  (its gated data and frozen features are read from `../../paper1_pre/`, never modified; the data
  gate `P2_D0_data_gate.py` SHA-256-checks them).

## Quick start

```bash
cd reproducibility_p2
python -m pip install -r requirements.txt      # tested with Python 3.12.3
make -f Makefile_P2 quick   # rebuild tables/figures from frozen CSVs (~2 min)
make -f Makefile_P2 all     # full rerun (~several hours)
make -f Makefile_P2 check   # sha256sum -c SHA256SUMS_P2.txt
```

Every number in the manuscript is a LaTeX macro generated from
`reproducibility_p2/tables_p2/` with its source in `manuscript_p2/numbers_sources.csv`.

## Data

All data are Paper 1's gated/third-party files (see `../paper1_pre/reproducibility_pre/README_P1.md`
for provenance, the 2023-05-24 volume repair, and redistribution terms). No raw vendor download is
redistributed; SHA-256 checksums and download scripts are provided.
