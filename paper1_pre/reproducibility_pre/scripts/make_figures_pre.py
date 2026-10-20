"""Regenerate every manuscript figure (Figs. 1-6 and C1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import figures_pre_lib as L
for f in L.ALL:
    f(); print("ok", f.__name__)
