import argparse
import json
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--fixtures', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--write', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(Path(args.source).resolve()))

import numpy as np
import kickscore as ks


def snapshot(model):
    result = {'last_t': model.last_t, 'items': {}}
    for name, item in model.item.items():
        fitter = item.fitter
        result['items'][name] = {key: np.asarray(getattr(fitter, key)).tolist() for key in ['ts', 'ms', 'vs', 'ns', 'xs']}
        result['items'][name]['pending'] = list(fitter.ts_new)
        result['items'][name]['fitted'] = fitter.is_fitted
        if fitter.is_fitted:
            result['items'][name]['prediction'] = [a.tolist() for a in item.predict(np.array([-1.0, 0.5, 3.0]))]
    return result


if args.write:
    fixtures = {}
    for fitter in ['recursive', 'batch']:
        for kind in ['binary', 'ternary', 'difference', 'count', 'countdiff']:
            for stage in ['unallocated', 'fitted', 'pending']:
                model = {'binary': ks.BinaryModel, 'ternary': ks.TernaryModel, 'difference': ks.DifferenceModel, 'count': ks.CountModel, 'countdiff': ks.CountDiffModel}[kind]()
                kernel = ks.kernel.Constant(0.5) + ks.kernel.Matern52(0.3, 2.0)
                for name in ['a', 'b']:
                    model.add_item(name, kernel, fitter=fitter)
                for i in range(3):
                    kwargs = {'t': float(i // 2)}
                    if kind == 'ternary':
                        kwargs['tie'] = i == 0
                    elif kind == 'difference':
                        kwargs['diff'] = 0.4
                    elif kind == 'count':
                        kwargs['count'] = i
                    elif kind == 'countdiff':
                        kwargs['diff'] = i - 1
                    model.observe({'a': 0.8}, ['b'], **kwargs)
                    if stage == 'pending' and i == 1:
                        model.fit(lr=0.3, max_iter=3, tol=0.0)
                if stage == 'fitted':
                    model.fit(lr=0.3, max_iter=3, tol=0.0)
                fixtures[f'{fitter}/{kind}/{stage}'] = pickle.dumps(model)
    Path(args.fixtures).write_bytes(pickle.dumps(fixtures))
else:
    fixtures = pickle.loads(Path(args.fixtures).read_bytes())

results = {}
for name, payload in fixtures.items():
    model = pickle.loads(payload)
    assert model.observations[0]._items[0] is model.item['a']
    assert model.item['a'].kernel is model.item['b'].kernel
    before = snapshot(model)
    model.fit(lr=0.3, max_iter=3, tol=0.0)
    results[name] = {'before': before, 'after': snapshot(model), 'likelihood': float(model.log_likelihood)}
Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=True))
print(f'Wrote {len(results)} pickle comparisons to {args.output}')
