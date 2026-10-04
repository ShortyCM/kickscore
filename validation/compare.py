import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('reference')
parser.add_argument('native')
args = parser.parse_args()
reference = json.loads(Path(args.reference).read_text())
native = json.loads(Path(args.native).read_text())
failures = []
maximum = [0.0, '']


def compare(a, b, path):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            failures.append(f'{path}: different keys or result type')
            return
        for key in a:
            compare(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            failures.append(f'{path}: different lengths or result type')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}/{i}')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)):
            failures.append(f'{path}: different result type')
            return
        if math.isnan(a) and math.isnan(b):
            return
        if a == b:
            return
        error = abs(a - b)
        if math.isfinite(error) and error > maximum[0]:
            maximum[:] = [error, path]
        if not math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-11):
            failures.append(f'{path}: reference={a!r}, native={b!r}')
    elif a != b:
        failures.append(f'{path}: reference={a!r}, native={b!r}')


compare(reference, native, '')
print(f'Maximum absolute difference: {maximum[0]:.17g} at {maximum[1]}')
print(f'Mismatches: {len(failures)}')
for failure in failures[:40]:
    print(failure)
raise SystemExit(bool(failures))
