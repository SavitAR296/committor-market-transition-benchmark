r"""Consistency gate for Paper 1 cross-references (make exits on failure).
1. Supplemental tables are numbered S1..Sn without gaps (supplement.aux) and the PDF captions read S1..Sn in order.
2. Every \Sref{key} in the main-text sources and every \Mref{key} in the supplement resolves.
3. Neither PDF contains '??' or 'S??'; every 'Table S<k>' quoted in main.pdf satisfies k <= n.
4. 'Tables S1--S<n>' quoted in the main text equals the number of supplemental tables."""
import re, subprocess, sys
from pathlib import Path
MS = Path(__file__).resolve().parents[2] / 'manuscript'
LAB = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}')
fail = []
supp = {k: v for k, v in LAB.findall((MS / 'supplement.aux').read_text()) if k.startswith('tab:S-')}
nums = sorted(int(v[1:]) for v in supp.values()); n = len(nums)
if nums != list(range(1, n + 1)): fail.append(f'supplemental table numbers not contiguous: {nums}')
txt = lambda pdf: subprocess.run(['pdftotext', '-layout', str(MS / pdf), '-'], capture_output=True, text=True).stdout
st = txt('supplement.pdf'); caps = [int(x) for x in re.findall(r'^\s*Table S(\d+):', st, re.M)]
if caps != list(range(1, n + 1)): fail.append(f'supplement captions read {caps}')
main_src = ''.join((MS / f).read_text() for f in ['main.tex'] + [p.name for p in MS.glob('sec*.tex')] + ['appendices.tex', 'refs.tex'])
for k in set(re.findall(r'\\Sref\{([^}]*)\}', main_src)):
    if k not in {kk[len('tab:S-'):] for kk in supp}: fail.append(f'\\Sref{{{k}}} has no supplemental table')
mlab = {k for k, _ in LAB.findall((MS / 'main.aux').read_text())}
for k in set(re.findall(r'\\Mref\{([^}]*)\}', (MS / 'supplement.tex').read_text() + ''.join(p.read_text() for p in (MS / 'tables').glob('S*.tex')))):
    if k not in mlab: fail.append(f'\\Mref{{{k}}} has no main-text label')
mt = txt('main.pdf')
for name, t in [('main.pdf', mt), ('supplement.pdf', st)]:
    if '??' in t: fail.append(f'{name} contains ??')
quoted = [int(x) for x in re.findall(r'Tables? S(\d+)', mt)] + [int(x) for x in re.findall(r'S\d+[\u2013-]S(\d+)', mt)]
if quoted and max(quoted) > n: fail.append(f'main.pdf quotes Table S{max(quoted)} but the supplement has {n}')
rng = re.findall(r'Tables S1[\u2013-]+S(\d+)', mt)
if rng and any(int(x) != n for x in rng): fail.append(f'main.pdf says Tables S1-S{rng} but there are {n}')
print(f'supplemental tables S1-S{n}; main.pdf quotes {sorted(set(quoted))}')
if fail:
    print('CHECK FAILED:\n  ' + '\n  '.join(fail)); sys.exit(1)
print('cross-reference check passed')
