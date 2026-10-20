# Zenodo: correcting the old record and publishing v2.0.0

Status of the Zenodo records (verified 2026-10-20 through the public API):

| record | value |
|---|---|
| version DOI | `10.5281/zenodo.22774226` |
| concept DOI | `10.5281/zenodo.22774225` |
| version | `v1.0.0-submission` |
| created | 2026-09-15 |
| creators | Xing, Boyan (sole author) |
| problem | description states "Status: submitted for publication (Chaos, Solitons and Fractals)" — **false**; the draft was never submitted. |

The record is older than 30 days, so it **cannot be deleted**; the correct procedure is (1) correct the
old record's metadata, and (2) publish a new version `v2.0.0`. Do **not** create a brand-new record:
the concept DOI must keep resolving to the corrected lineage, so that links already printed with the
old DOI land on the corrected version.

## Step 1 — correct the old record (5 minutes, keeps the old DOI)

Log in to Zenodo with the account that owns the record, open `https://zenodo.org/records/22774226`,
click *Edit*, and change the **Description** to the text below (append it above or replace the old
description). Change the title if desired (optional). Save and republish. Nothing else in Step 1
affects the DOI.

> **Correction notice (2026-10-20).** The draft archived in this record, *Committor benchmarks for
> market regime transitions: state information, rolling-window failure, and out-of-sample limits*,
> was **never submitted to any journal**. The statement "Status: submitted for publication (Chaos,
> Solitons and Fractals)" in this record's description was erroneous. The draft is superseded by two
> companion manuscripts by Boyan Xing and Jiayi Pan — *Observer-induced irreversibility in
> trailing-window early-warning states* (prepared for submission to Physical Review E) and *What
> does a committor score forecast? Hitting-probability warnings of market stress, their target and
> their evaluation* (prepared for submission to the International Journal of Forecasting) — whose
> reproducibility archive is deposited as **version 2.0.0** of this record and at
> https://github.com/SavitAR296/committor-market-transition-benchmark. The frozen draft contents of
> this version are preserved for provenance.

## Step 2 — publish version 2.0.0

Either route works; route A is automatic if the GitHub-Zenodo integration is enabled for the
repository.

### Route A — GitHub release (automatic if integration is enabled)

1. Push the repository and tag `v2.0.0`. Paper 1's manuscript cites commit `b9fa527…` (the commit
   that froze the reproducibility state); the release tag `v2.0.0` sits on the following commit
   `dc69cca…`, which differs from the cited commit only by the filled commit identifier and the
   regenerated SHA-256 manifest. Either commit is immutable and resolves; archive the **tagged**
   tree (`v2.0.0`).
2. On GitHub, create a release from the tag `v2.0.0` with title `v2.0.0 — two-paper
   reproducibility archive` and paste the description of Step 2B below.
3. If the Zenodo GitHub app was enabled for this repository (Settings → Integrations → Zenodo), the
   release automatically creates the new Zenodo version, which you then review and publish on
   Zenodo. If the integration is not enabled, use Route B.

### Route B — manual deposit

1. On Zenodo, open the old record page and click **New version** (under the version list).
2. Upload the archive of the repository at the release commit (e.g. GitHub's
   *Download ZIP* of tag `v2.0.0`, or `git archive v2.0.0 | tar -x -C <dir>` on your machine, zipped).
3. Fill the metadata:

   - **Title**: Committor market-transition benchmark: reproducibility archive for two companion
     papers (version 2.0.0)
   - **Version**: 2.0.0
   - **Creators**: Boyan Xing (Department of Mathematical Sciences, Faculty of Science and
     Technology, Beijing Normal-Hong Kong Baptist University, Zhuhai, Guangdong, China); Jiayi Pan
     (Department of Accounting, Faculty of Business and Management, Beijing Normal-Hong Kong Baptist
     University, Zhuhai, Guangdong, China). Add ORCIDs if available.
   - **Description**: paste the block below.
   - **License**: MIT.
   - **Upload type**: Software.
   - **Keywords**: early warning signals; time irreversibility; committor; hitting probability;
     forecast evaluation; model confidence set; reproducibility.

   > Reproducibility archive for two companion manuscripts by Boyan Xing and Jiayi Pan. (1)
   > *Observer-induced irreversibility in trailing-window early-warning states*, prepared for
   > submission to Physical Review E (directory `paper1_pre/`). (2) *What does a committor score
   > forecast? Hitting-probability warnings of market stress, their target and their evaluation*,
   > prepared for submission to the International Journal of Forecasting (double-blind; its package
   > is provided as anonymous supplementary material and will be deposited here upon acceptance).
   > Data provenance and redistribution terms, SHA-256 manifests, scripts and seeds are described in
   > the repository README. This version supersedes v1.0.0 (the never-submitted single-author draft;
   > see the correction notice on that version).

4. **Publish**. Zenodo issues the new version DOI, e.g. `10.5281/zenodo.XXXXXXXX`.

## Step 3 — backfill the new DOI (after Step 2 only)

- `paper1_pre/manuscript/main.tex`: set `\ZenodoDOI` to the **new version DOI** (not the concept
  DOI); `\RepoCommit` is already set to the release commit.
- `CITATION.cff`: replace the `doi:` line with the new version DOI (currently the concept DOI with a
  comment).
- `README.md`: replace the two "DOI filled at deposit" placeholders with the new version DOI.
- After acceptance of either paper, add the journal DOI to `CITATION.cff` and to the manuscript data
  availability paragraphs.

## Why not delete the old record

Zenodo only lets owners delete within 30 days of upload; after that, deletion requires a support
request and leaves a tombstone page. A correction note is more transparent for reviewers and keeps
the DOI lineage intact. The erroneous "Submitted to CSF" text no longer appears anywhere in the
repository; the archived draft is marked as never submitted (`archive_csf_draft/README.md`).
