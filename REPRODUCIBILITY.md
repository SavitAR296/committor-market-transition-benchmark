# Reproducibility package - Paper B

This directory contains the data snapshots, executable CSF-revision code, frozen forecast/result CSVs, and vector figures used by the submission **Committor benchmarks for market regime transitions: state information, rolling-window failure, and out-of-sample limits**.

## Data
The archived project uses public Bitcoin, Ether, S&P 500 and VIX snapshots. The matched real-data attribution in the final paper is run on `tables/features_SPX.csv`; BTC and ETH are used only for archived cross-market dynamical diagnostics in the final manuscript.

## Environment
The new headline CSF experiments were checked with Python 3.13.5 and the package versions in `requirements.txt`. The included headline scripts do not require JAX, optax or hmmlearn.

## Headline workflow
From this directory:

```bash
python -m pip install -r requirements.txt
python experiments/F_analytic.py  # regenerates the analytic double-well benchmark (figures/fig3_prop2_double_well.pdf)
python experiments/B_csf_additions.py all --reps 8
python experiments/B_inference_fast.py
python experiments/make_figures.py
```

`B_csf_additions.py` contains the strict annual expanding-window feature attribution, the synthetic double-well observation-design phase diagram, and target-set sensitivity. `B_inference_fast.py` recomputes moving-block, stationary and contiguous-segment uncertainty for matched Brier-skill comparisons. Frozen outputs used in the manuscript are included so the paper can be rebuilt without rerunning the heavier annual fits.

## Frozen headline outputs
- `tables_csf/B_feature_attribution_forecasts.csv`
- `tables_csf/B_feature_attribution_metrics.csv`
- `tables_csf/B_synthetic_phase.csv`
- `tables_csf/B_target_sensitivity.csv`
- `tables_csf/B_inference_robustness.csv`
- `tables_csf/B_pca_loadings.csv`

Archived real-data dynamical diagnostics are in `tables/E2v2_diag_*.csv`. Vector versions of the final figures are included in `figures_csf/`. The final `B_fig1_doublewell.pdf` is byte-identical to `figures/fig3_prop2_double_well.pdf` produced by `F_analytic.py`; rename/copy that output when rebuilding the submission figure folder.

## Evaluation protocol
The final feature-attribution experiment uses annual expanding windows. The OU hitting score is sequentially recalibrated using only prior out-of-sample predictions/labels when available; the first evaluable period falls back to the training climatology. Baselines are matched to the same information set. The synthetic phase diagram separates latent-state information from rolling-coordinate compression and estimator error.

## Randomness
Synthetic experiments use deterministic seeds beginning with `20260914`; the dependence bootstrap uses seed `20260915`. Scikit-learn model random states are fixed in the archived script.

## AI-assisted code development
OpenAI ChatGPT (GPT-5.6 Sol; OpenAI, accessed September 2026) assisted with drafting/refactoring matched-feature, synthetic, bootstrap and plotting code. All reported numerical outputs were produced by executable code against the supplied data or frozen forecast files and checked against the manuscript tables. Authors remain responsible for validation before submission.

## Before public deposition
Add the final repository URL, immutable commit hash, license, and archival DOI/Zenodo record. These identifiers are intentionally left as author-side submission fields.
