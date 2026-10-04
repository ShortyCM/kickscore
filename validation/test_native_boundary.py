import pickle

import numpy as np
import pytest

import kickscore as ks
from kickscore.fitter import BatchFitter, RecursiveFitter


@pytest.mark.parametrize('fitter_type', [BatchFitter, RecursiveFitter])
def test_writable_native_state_and_growth(fitter_type):
    fitter = fitter_type(ks.kernel.Constant(0.5) + ks.kernel.Matern52(0.8, 2.0))
    fitter.add_sample(0.0)
    fitter.allocate()
    xs = fitter.xs
    xs[:] = 0.4
    fitter.ns[:] = 0.2
    fitter.fit()
    original = np.asarray(fitter.ms).copy()
    fitter.add_sample(1.0)
    fitter.allocate()
    assert xs[0] == 0.4
    assert len(xs) == 2
    assert fitter.ms[0] == original[0]
    xs[1] = 0.3
    fitter.ns[1] = -0.1
    fitter.fit()
    restored = pickle.loads(pickle.dumps(fitter))
    for a, b in zip(fitter.predict(np.array([-0.1, 0.5, 2.0])), restored.predict(np.array([-0.1, 0.5, 2.0]))):
        np.testing.assert_array_equal(a, b)


def test_pickle_keeps_observation_fitter_connections():
    model = ks.TernaryModel(obs_type='logit')
    kernel = ks.kernel.Constant(0.5) + ks.kernel.Exponential(0.3, 2.0)
    for name in ['a', 'b']:
        model.add_item(name, kernel)
    model.observe(['a'], ['b'], t=0.0, tie=True)
    model.fit(lr=0.3, max_iter=3, tol=0.0)
    restored = pickle.loads(pickle.dumps(model))
    for value in [model, restored]:
        value.observe(['b'], ['a'], t=1.0)
        value.fit(lr=0.3, max_iter=3, tol=0.0)
    np.testing.assert_array_equal(model.item['a'].scores[1], restored.item['a'].scores[1])
    assert restored.observations[0]._items[0] is restored.item['a']


def test_python_override_is_explicitly_rejected():
    model = ks.BinaryModel()
    model.add_item('a', ks.kernel.Constant(1.0))
    model.observe(['a'], [], 0.0)
    model.observations[0].ep_update = lambda lr: 0.0
    with pytest.raises(NotImplementedError, match='Python overrides'):
        model.fit()
    assert not model.item['a'].fitter.is_fitted
