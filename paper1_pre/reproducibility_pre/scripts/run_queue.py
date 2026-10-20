"""Resumable driver for the 300-s-per-command sandbox: runs the remaining scripts in order, each with the remaining
time budget (env PRE_BUDGET); a script marks completion with tables_pre/_cache/DONE_<name>.  `make all` does not need
this driver (it simply runs every script)."""
import os, sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
QUEUE = ['E4_surrogates', 'S4_doublewell_collapse', 'S5_doublewell_twowindow', 'S3_ratio_definition',
         'E5_murphy_inference', 'E6_score_drift', 'E7_cases', 'E8_pca_geometry']
deadline = time.time() + float(os.environ.get('QUEUE_SECONDS', '260'))
for name in QUEUE:
    if (ROOT / 'tables_pre' / '_cache' / f'DONE_{name}').exists() or not (ROOT / 'scripts' / f'{name}.py').exists():
        continue
    remaining = deadline - time.time()
    if remaining < 40:
        print('queue paused (time)'); break
    env = dict(os.environ, PRE_BUDGET=str(max(remaining - 75, 5)))
    try:
        r = subprocess.run([sys.executable, str(ROOT / 'scripts' / f'{name}.py')], env=env, timeout=remaining,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        print(name, 'timed out (cached units kept)'); break
    print(f'== {name} rc={r.returncode}\n' + '\n'.join(l for l in out.splitlines() if 'arning' not in l)[-2500:])
    if r.returncode != 0 or not (ROOT / 'tables_pre' / '_cache' / f'DONE_{name}').exists():
        break
