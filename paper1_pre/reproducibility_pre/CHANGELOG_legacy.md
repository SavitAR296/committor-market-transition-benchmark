# Changes to the legacy CSF package (`../reproducibility/`)

Frozen data and frozen result files (data/, tables/, tables_csf/, figures_csf/) are byte-identical to the CSF
submission package. Only documentation and one argparse default were changed (outline Sec. 9 items 1-2; Sec. 10 #6):

* `README.md`: `--reps 8` -> `--reps 6` (the frozen phase diagram has 6 replications per tau_c; S4 re-generated it
  with the frozen seed sequence and identified the 6-replication ordering, reproducing every AUC/BSS to 1e-16);
  the pointer to the non-shipped `figures/fig3_prop2_double_well.pdf` is explained (Fig. C1 of the PRE manuscript).
* `experiments/B_csf_additions.py`: default `--reps` and `synthetic_phase_diagram(reps=...)` 8 -> 6.
* `experiments/F_analytic.py`, `src/committor.py`: docstrings no longer refer to the CSF-era Propositions 2/3,
  Lemma 1 or Fig. 4; CIR/Hawkes helpers are marked as companion-study code not used in the PRE manuscript.
* `SHA256SUMS.txt` regenerated (now also lists itself-excluded files created on first run, if present).

| file | old sha256 | new sha256 |
|---|---|---|
| ./README.md | f3399b60dc4fd62b... | 8e6a6646dd7e1cc5... |
| ./experiments/B_csf_additions.py | 7a326962d755e8ee... | 4ac9e79484b70b52... |
| ./experiments/F_analytic.py | 1ab43b3582f64b56... | 50059781108cbaaf... |
| ./src/committor.py | a73fe62138cb75d1... | 28f9374109251ac5... |
