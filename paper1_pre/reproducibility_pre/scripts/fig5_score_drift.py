"""Generate figures_pre/fig5_score_drift.pdf/.png from tables_pre CSVs (see src/figures_pre_lib.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import figures_pre_lib as L
L.fig5_score_drift()
