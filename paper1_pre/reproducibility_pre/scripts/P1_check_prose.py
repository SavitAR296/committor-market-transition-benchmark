"""Lists decimal numbers typed by hand in the main-text sections (not produced by P1_numbers.py).  Informational:
model parameters and definitions are expected here; result numbers should be macros."""
import re
from pathlib import Path
MS = Path(__file__).resolve().parents[2] / 'manuscript'
for f in sorted(MS.glob('sec*.tex')):
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if line.lstrip().startswith('%'): continue
        clean = re.sub(r'\\(label|ref|eqref|cite|Sref|Mref|input|includegraphics)\{[^}]*\}', '', line)
        hits = re.findall(r'(?<![\w.])[-+]?\d+\.\d+', clean)
        if hits: print(f'{f.name}:{i}: {", ".join(hits)}')
