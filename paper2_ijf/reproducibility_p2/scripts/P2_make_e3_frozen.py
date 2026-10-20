"""P2_make_e3_frozen -- writes src/p2_e3_frozen.py: verbatim copies (AST extraction, no import) of four helpers of the
frozen script reproducibility_pre/scripts/E3_matched_estimators.py.  Deterministic; rerun only if E3 changes."""
import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import P2, PRE

src_path = PRE / 'scripts' / 'E3_matched_estimators.py'
src = src_path.read_text(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
want = ['target_direction', 'msm_scores', 'amc_scores', 'gbt_cv']
chunks = [''.join(lines[n.lineno - 1:n.end_lineno]) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
header = f'''"""Verbatim copies of four helpers of the frozen Paper-B script reproducibility_pre/scripts/E3_matched_estimators.py
(sha256 {sha}): target_direction, msm_scores, amc_scores, gbt_cv, with the constants they use.  Copied rather than
imported because importing E3 creates a cache directory inside the read-only Paper 1 tree.  P2_F1_forecasts.py --check
verifies that the forecasts on the pre-repair feature table reproduce the old E3 forecasts exactly at stored precision.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

N_MSM, K_AMC = 100, 20
GRID = [(1, 120), (2, 60), (2, 120), (2, 240), (3, 120)]          # (max_depth, n_estimators)


'''
(P2 / 'src' / 'p2_e3_frozen.py').write_text(header + '\n\n'.join(chunks))
print('wrote src/p2_e3_frozen.py from E3 sha256', sha)
