# Committor Market-Transition Benchmark

Reproducibility repository for the manuscript:

**Committor benchmarks for market regime transitions: state information, rolling-window failure, and out-of-sample limits**

Boyan Xing — Department of Mathematical Sciences, Faculty of Science and Technology, Beijing Normal-Hong Kong Baptist University, Zhuhai, Guangdong, China.

Submitted to *Chaos, Solitons & Fractals*.

## What this repository contains

- `src/` — finite-difference committor/hitting solver, daily data utilities, crisis-label construction and evaluation code.
- `experiments/` — analytic double-well check (`F_analytic.py`), all CSF additions (`B_csf_additions.py`), dependence-aware inference (`B_inference_fast.py`) and figure generation (`make_figures.py`).
- `data/` — public Bitcoin, Ether, S&P 500 and VIX daily snapshots.
- `tables/` and `tables_csf/` — frozen forecast, metrics, synthetic phase-diagram and bootstrap CSV outputs used by every headline table.
- `figures_csf/` — final vector and raster figures.

## Reproduce the headline results

Tested with Python 3.13. The headline scripts do not require JAX, optax or hmmlearn.

```bash
python -m pip install -r requirements.txt
python experiments/F_analytic.py
python experiments/B_csf_additions.py all --reps 8
python experiments/B_inference_fast.py
python experiments/make_figures.py
```

Frozen outputs used in the manuscript are included so the paper can be rebuilt without rerunning the heavier annual fits.

## Locked headline results

- Nonlinear synthetic benchmark (tau_c = 20): oracle mean AUC/BSS ≈ 0.894/0.551; rolling-flexible state at w = 80 falls to ≈ 0.805/0.331.
- Information loss persists without slow control drift (w = 1 → 80: AUC 0.867 → 0.750).
- S&P common-feature state: raw OU AUC ≈ 0.901 (5 d) and 0.776 (20 d); strict sequential calibration gives BSS ≈ 0.077 and 0.005.
- Same-state supervised baselines often match or exceed the dynamical score; no universal physics-over-ML advantage is claimed.
- Target-set sensitivity is modest; dependence-aware bootstrap inference weakens small score differences.

## Evaluation protocol

The feature-attribution experiment uses annual expanding windows. The OU hitting score is sequentially recalibrated using only prior out-of-sample predictions/labels, falling back to the training climatology when history is insufficient. Baselines are matched to exactly the same information set. The synthetic phase diagram separates latent-state information from rolling-coordinate compression and estimator error.

## Randomness

Synthetic experiments use deterministic seeds beginning with `20260914`; the dependence bootstrap uses seed `20260915`. Scikit-learn random states are fixed in the archived scripts.

## AI-assisted code development

OpenAI ChatGPT (GPT-5.6 Sol; OpenAI, accessed September 2026) assisted with drafting/refactoring matched-feature, synthetic, bootstrap and plotting code. All reported numerical outputs were produced by executable code against the supplied data or frozen forecast files and checked against the manuscript tables. The author retains full responsibility.

## License

Code is released under the MIT License (see `LICENSE`). See `DATA_REDISTRIBUTION_NOTE.md` for the provenance and redistribution terms of the bundled market-data snapshots.

## Citation

See `CITATION.cff`. Once the manuscript is accepted, update this file and the repository description with the journal DOI.
