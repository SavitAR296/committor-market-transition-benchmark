# PRE rebuild of Paper B.  `make frozen` rebuilds every table and figure from frozen inputs and the CSVs in
# tables_pre/ (about 2 min); `make all` recomputes every CSV first (about 90 min on one core); `make check` verifies
# the SHA256 manifest.  Scripts are resumable: units are cached in tables_pre/_cache (delete it to recompute).
PY ?= python3
COMPUTE = T1_ledger_check T2_gaussian_observer T3_oneway_driving E1_extend_data E2_event_log E3_matched_estimators \
          E4_surrogates E4b_leverage S4_doublewell_collapse S4b_effective_variance S4c_scaling_fit \
          S5_doublewell_twowindow S3_ratio_definition E5_murphy_inference E6_score_drift E7_cases E8_pca_geometry
.PHONY: frozen all check manuscript
frozen:
	$(PY) scripts/T1_ledger_check.py
	$(PY) scripts/make_tables_pre.py
	$(PY) scripts/make_figures_pre.py
all:
	@for s in $(COMPUTE); do echo "== $$s"; PRE_BUDGET=1e9 $(PY) scripts/$$s.py || exit 1; done
	$(MAKE) frozen
check:
	sha256sum -c SHA256SUMS.txt
manuscript:
	cd ../manuscript && latexmk -pdf main.tex && latexmk -pdf supplement.tex
