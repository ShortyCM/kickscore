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
    assert len(xs) == 1
    assert len(fitter.xs) == 2
    assert fitter.ms[0] == original[0]
    fitter.xs[1] = 0.3
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


@pytest.mark.parametrize('fitter_type', [BatchFitter, RecursiveFitter])
def test_numpy_views_write_native_state(fitter_type):
    fitter = fitter_type(ks.kernel.Constant(1.0))
    fitter.add_sample(0.0)
    fitter.allocate()
    values = fitter.xs
    assert type(values) is np.ndarray
    np.asarray(values)[:] = 0.5
    values[:].fill(0.75)
    assert fitter.xs[0] == 0.75
    assert np.shares_memory(values, fitter.xs)
    fitter.ns[:] = 0.25
    fitter.fit()
    np.testing.assert_allclose(fitter.ms, [0.25 / 1.75], rtol=2e-10, atol=2e-11)
    before = values.copy()
    fitter.add_sample(1.0)
    fitter.allocate()
    values[:] = 99.0
    np.testing.assert_array_equal(fitter.xs[:1], before)


@pytest.mark.parametrize("kernel", [
    ks.kernel.Exponential(0.3, 2.0),
    ks.kernel.Matern32(0.3, 2.0),
    ks.kernel.Matern52(0.3, 2.0),
    ks.kernel.Constant(0.5) + ks.kernel.Matern32(0.3, 2.0) + ks.kernel.Matern52(0.4, 1.0),
    ks.kernel.PiecewiseConstant(0.5, np.array([0.5])),
    ks.kernel.Wiener(0.2, 0.0, 0.5),
])
def test_simulation_repeated_times(kernel):
    values = kernel.simulate(np.array([1.0, 0.0, 0.0, 1.0]))
    assert values.shape == (4,)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[::2], values[1::2])


def test_piecewise_bounds_are_writable():
    kernel = ks.kernel.PiecewiseConstant(1.0, np.array([1.0]))
    kernel.bounds[:] = 2.0
    assert kernel.k_mat(np.array([1.5]), np.array([0.0]))[0, 0] == 1.0
