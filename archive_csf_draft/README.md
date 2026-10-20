# Archive: the never-submitted single-author draft (v1.0.0)

This directory preserves the top-level documentation of an earlier single-author draft,

> *Committor benchmarks for market regime transitions: state information, rolling-window failure,
> and out-of-sample limits* (B. Xing, September 2026),

which was **never submitted to any journal**. Statements in the archived files referring to a
"submission" or to *Chaos, Solitons & Fractals* are historical and **erroneous**; they describe a
draft that was abandoned. The Zenodo record `10.5281/zenodo.22774226` that carried the same wording
has been corrected and superseded by version 2.0.0 (see `../ZENODO_DEPOSIT.md`).

The draft's frozen code, data, tables and figures are preserved **byte-identical** in the read-only
legacy package `../paper1_pre/reproducibility/` (SHA-256-manifested by
`../paper1_pre/reproducibility_pre/SHA256SUMS_P1.txt`), because Paper 1's pipeline consumes several
frozen inputs from it. The draft itself is not part of either current manuscript.

## Files

- `REPRODUCIBILITY_legacy.md` — the draft's reproducibility documentation (historical; includes the
  draft's AI-use disclosure and the pre-publication notes that no longer apply).
- `DATA_REDISTRIBUTION_NOTE_legacy.md` — the draft's data-provenance and redistribution policy note;
  the policy (no redistribution of raw vendor files, download scripts + checksums instead) is
  carried over verbatim into the current repository.
- `SHA256SUMS_legacy.txt` — SHA-256 manifest of the draft's root package (superseded by the manifests
  in `../paper1_pre/`).
- `requirements_legacy.txt` — environment of the draft package (Python 3.13).
- `CITATION_cff_v1.0.0.txt` — the draft's citation metadata, kept as a historical record; the current
  metadata is `../CITATION.cff`.

Everything in this directory is documentation. The current manuscripts supersede the draft, and the
root `../README.md` states their status.
