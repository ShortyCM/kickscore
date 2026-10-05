import argparse
import json
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--stage', choices=['kernels', 'moments', 'models'], required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.source).resolve()))

import numpy as np
import kickscore as ks
from kickscore.observation import GaussianObservation, LogitTieObservation, LogitWinObservation, PoissonObservation, ProbitTieObservation, ProbitWinObservation, SkellamObservation


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(val) for val in value]
    if hasattr(value, '__array__'):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attempt(call):
    try:
        return plain(call())
    except Exception as error:
        return {'error': type(error).__name__, 'message': str(error)}


def kernels():
    return {
        'constant': ks.kernel.Constant(0.7),
        'exponential': ks.kernel.Exponential(0.8, 1.7),
        'matern32': ks.kernel.Matern32(0.8, 1.7),
        'matern52': ks.kernel.Matern52(0.8, 1.7),
        'affine': ks.kernel.Affine(0.7, 0.1, -2.0),
        'wiener': ks.kernel.Wiener(0.1, -2.0, 0.7),
        'piecewise': ks.kernel.PiecewiseConstant(0.7, np.array([0.5, 2.0])),
        'periodic': ks.kernel.PeriodicExponential(0.8, 1.7, 3.0),
        'add_exp': ks.kernel.Constant(0.7) + ks.kernel.Exponential(0.8, 1.7),
        'add32': ks.kernel.Constant(0.7) + ks.kernel.Matern32(0.8, 1.7),
        'add52': ks.kernel.Constant(0.7) + ks.kernel.Matern52(0.8, 1.7),
        'add_matern_pair': ks.kernel.Matern32(0.3, 1.1) + ks.kernel.Matern52(0.4, 2.1),
        'add_all_matern': ks.kernel.Exponential(0.2, 1.7) + ks.kernel.Matern32(0.3, 1.1) + ks.kernel.Matern52(0.4, 2.1),
        'add_general': ks.kernel.Constant(0.7) + ks.kernel.Matern32(0.3, 1.1) + ks.kernel.Matern52(0.4, 2.1) + ks.kernel.Matern52(0.2, 0.9),
    }


result = {}
if args.stage == 'kernels':
    times = np.array([-1.0, 0.0, 0.5, 0.5, 1.7, 2.0, 3.1])
    for name, kernel in kernels().items():
        record = {'matrix': kernel.k_mat(times), 'diag': kernel.k_diag(times)}
        if name != 'periodic':
            record['h'] = kernel.measurement_vector
            record['states'] = [kernel.state_cov(t) for t in times]
            record['transitions'] = [kernel.transition(0.0, t) for t in [0.0, 1e-6, 0.1, 2.0, 20.0]]
            record['noise'] = [kernel.noise_cov(0.0, t) for t in [0.0, 1e-6, 0.1, 2.0, 20.0]]
        result[name] = record
elif args.stage == 'moments':
    item = ks.Item(ks.kernel.Constant(1.0), 'recursive')
    observations = [ProbitWinObservation([(item, 1.0)], 0.0, 0.1), LogitWinObservation([(item, 1.0)], 0.0, 0.1), ProbitTieObservation([(item, 1.0)], 0.0, 0.1), LogitTieObservation([(item, 1.0)], 0.0, 0.1), GaussianObservation([(item, 1.0)], 0.4, 0.8, 0.0), PoissonObservation([(item, 1.0)], 2, 0.0), SkellamObservation([(item, 1.0)], -2, 0.2, 0.0)]
    for obs in observations:
        for mean in [-20.0, -2.0, 0.0, 2.0, 20.0]:
            for var in [0.01, 0.5, 3.0]:
                for method in ['match_moments', 'cvi_expectations']:
                    key = f'{type(obs).__name__}/{mean}/{var}/{method}'
                    result[key] = attempt(lambda: getattr(obs, method)(mean, var))
else:
    for name, kernel in kernels().items():
        for fitter in ['recursive', 'batch']:
            if name == 'periodic' and fitter == 'recursive':
                continue
            for model_name in ['probit', 'logit', 'probit_tie', 'logit_tie', 'difference', 'count', 'countdiff']:
                for method in ['ep', 'kl']:
                    if model_name == 'difference' and method == 'kl':
                        continue
                    if model_name in ['probit', 'logit']:
                        model = ks.BinaryModel(model_name)
                    elif model_name.endswith('_tie'):
                        model = ks.TernaryModel(0.2, model_name.split('_')[0])
                    else:
                        model = {'difference': ks.DifferenceModel, 'count': ks.CountModel, 'countdiff': ks.CountDiffModel}[model_name]()
                    for player in ['a', 'b', 'c', 'unused']:
                        model.add_item(player, kernel, fitter=fitter)
                    records = []
                    for stage in range(2):
                        for i in range(stage * 4, stage * 4 + 4):
                            first, second = {'a': 0.7, 'c': 0.3}, {'b': 1.0}
                            if i % 2:
                                first, second = second, first
                            t = float(i // 2)
                            if model_name.endswith('_tie'):
                                model.observe(first, second, t=t, tie=i % 3 == 0)
                            elif model_name in ['probit', 'logit']:
                                model.observe(first, second, t=t)
                            elif model_name == 'difference':
                                model.observe(first, second, diff=0.4, t=t)
                            elif model_name == 'count':
                                model.observe(first, second, count=i % 3, t=t)
                            else:
                                model.observe(first, second, diff=i % 3 - 1, t=t)
                        record = {'fit': attempt(lambda: model.fit(method=method, lr=0.3, tol=0.0, max_iter=4))}
                        record['state'] = {player: {attr: getattr(item.fitter, attr) for attr in ['ts', 'ms', 'vs', 'ns', 'xs']} for player, item in model.item.items()}
                        record['prediction'] = attempt(lambda: model.item['a'].predict(np.array([-1.0, 0.0, 0.5, 1.0, 4.0])))
                        record['likelihood'] = attempt(lambda: model.log_likelihood)
                        record['probabilities'] = attempt(lambda: model.probabilities({'a': 0.7, 'c': 0.3}, ['b'], t=4.0))
                        record['pickle'] = attempt(lambda: pickle.loads(pickle.dumps(model)).item['a'].predict(np.array([4.0])))
                        records.append(plain(record))
                    result[f'{name}/{fitter}/{model_name}/{method}'] = records
Path(args.output).write_text(json.dumps(plain(result), indent=2, allow_nan=True))
print(f'{args.stage}: wrote {len(result)} cases to {args.output}')
