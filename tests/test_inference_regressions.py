from math import exp, log, pi, sqrt
from typing import Literal

import numpy as np
import pytest
from scipy.integrate import quad

import kickscore as ks
from kickscore.fitter import RecursiveFitter
from kickscore.fitter._kernel_cache import KernelCache
from kickscore.observation._batch import ObservationBatch
from kickscore.observation.ordinal import _mm_logit_win, _mm_probit_tie


@pytest.mark.parametrize("mean", [-100.0, -20.0, 0.0, 20.0, 100.0])
@pytest.mark.parametrize("var", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("margin", [0.1, 2.0])
def test_probit_tie_against_quadrature(mean: float, var: float, margin: float):
    total_var = 1 + var
    density = lambda x: exp((2 * mean * x - x * x) / (2 * total_var))
    integral = quad(density, -margin, margin, epsabs=1e-12, epsrel=1e-12)[0]
    first = quad(lambda x: x * density(x), -margin, margin)[0] / integral
    second = quad(lambda x: x * x * density(x), -margin, margin)[0] / integral
    expected = (
        -mean * mean / (2 * total_var) - log(sqrt(2 * pi * total_var)) + log(integral),
        (first - mean) / total_var,
        (second - first * first) / total_var**2 - 1 / total_var,
    )
    np.testing.assert_allclose(_mm_probit_tie(mean, var, margin), expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("mean", [-1000.0, -100.0, 100.0, 1000.0])
def test_logit_extreme_finite_inputs(mean: float):
    actual = _mm_logit_win(mean, 1.0)
    logpart = 0.5 - abs(mean)
    expected = (logpart, 1.0, 0.0) if mean < 0 else (np.log1p(-exp(logpart)), 0.0, 0.0)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_probit_zero_margin_prediction():
    model = ks.TernaryModel(margin=0.0)
    model.add_item("a", ks.kernel.Constant(1.0))
    model.add_item("b", ks.kernel.Constant(1.0))
    assert model.probabilities(["a"], ["b"], t=0.0) == (0.5, 0.0, 0.5)
    with pytest.raises(ValueError, match="margin"):
        _mm_probit_tie(0.0, 1.0, 0.0)


@pytest.mark.parametrize("method", ["ep", "kl"])
def test_zero_variance_cannot_report_convergence(method: Literal["ep", "kl"]):
    model = ks.BinaryModel()
    model.add_item("a", ks.kernel.Constant(0.0))
    model.add_item("b", ks.kernel.Constant(1.0))
    model.observe(["a"], ["b"], t=0.0)
    with pytest.raises(FloatingPointError):
        model.fit(method=method)
    with pytest.raises(RuntimeError):
        model.probabilities(["a"], ["b"], t=1.0)


@pytest.mark.parametrize("method", ["ep", "kl"])
def test_nonfinite_custom_update_cannot_report_convergence(method: Literal["ep", "kl"]):
    model = ks.BinaryModel()
    model.add_item("a", ks.kernel.Constant(1.0))
    model.observe(["a"], [], t=0.0)
    setattr(model.observations[0], method + "_update", lambda lr: float("nan"))
    with pytest.raises(FloatingPointError, match="convergence"):
        model.fit(method=method)
    assert not model.item["a"].fitter.is_fitted


def test_nonfinite_fitter_output_cannot_report_convergence(monkeypatch: pytest.MonkeyPatch):
    model = ks.BinaryModel()
    model.add_item("a", ks.kernel.Constant(1.0))
    model.observe(["a"], [], t=0.0)
    fitter = model.item["a"].fitter

    def corrupt_fit():
        fitter.ms[:] = np.nan
        fitter.is_fitted = True

    monkeypatch.setattr(fitter, "fit", corrupt_fit)
    with pytest.raises(FloatingPointError, match="fitted"):
        model.fit(tol=100.0)
    assert not fitter.is_fitted


def make_model(
    obs_type: Literal["probit", "logit"], ternary: bool, fitter: Literal["batch", "recursive"]
):
    model = ks.TernaryModel(obs_type=obs_type) if ternary else ks.BinaryModel(obs_type=obs_type)
    kernel = ks.kernel.Constant(0.5) + ks.kernel.Matern32(0.4, 2.0)
    for name in "ABCDE":
        model.add_item(name, kernel, fitter=fitter)
    for j in range(12):
        winners, losers = ({"A": 0.5, "B": 1.0}, {"C": 0.7, "D": 1.0})
        if j % 3 == 0:
            winners, losers = losers, winners
        if isinstance(model, ks.TernaryModel):
            model.observe(winners, losers, t=float(j // 3), tie=j % 4 == 0)
        else:
            model.observe(winners, losers, t=float(j // 3))
    return model


@pytest.mark.parametrize("obs_type", ["probit", "logit"])
@pytest.mark.parametrize("ternary", [False, True])
@pytest.mark.parametrize("method", ["ep", "kl"])
@pytest.mark.parametrize("fitter", ["recursive", "batch"])
def test_compiled_updates_match_scalar_path(
    monkeypatch: pytest.MonkeyPatch,
    obs_type: Literal["probit", "logit"],
    ternary: bool,
    method: Literal["ep", "kl"],
    fitter: Literal["batch", "recursive"],
):
    compiled = make_model(obs_type, ternary, fitter)
    scalar = make_model(obs_type, ternary, fitter)
    for step in range(2):
        if step:
            for model in (compiled, scalar):
                model.observe(["A", "C"], ["B", "D"], t=4.0)
        compiled.fit(method=method, lr=0.3, max_iter=5, tol=0.0)
        with monkeypatch.context() as context:
            context.setattr(ObservationBatch, "create", lambda observations, fitters: None)
            scalar.fit(method=method, lr=0.3, max_iter=5, tol=0.0)
        for name in compiled.item:
            a, b = compiled.item[name].fitter, scalar.item[name].fitter
            for attr in ("ms", "vs", "ns", "xs"):
                np.testing.assert_allclose(
                    getattr(a, attr), getattr(b, attr), rtol=2e-10, atol=2e-11
                )
        for a, b in zip(compiled.observations, scalar.observations):
            for attr in ("_logpart", "_exp_ll", "_ns_cav", "_xs_cav"):
                np.testing.assert_allclose(
                    getattr(a, attr), getattr(b, attr), rtol=2e-10, atol=2e-11
                )
        np.testing.assert_allclose(
            compiled.probabilities(["A", "B"], ["C", "D"], t=5.0),
            scalar.probabilities(["A", "B"], ["C", "D"], t=5.0),
            rtol=2e-10,
            atol=2e-11,
        )
        if method == "ep" or fitter == "recursive":
            np.testing.assert_allclose(compiled.log_likelihood, scalar.log_likelihood, rtol=2e-10)


@pytest.mark.parametrize(
    "kernel",
    [
        ks.kernel.Constant(1.0) + ks.kernel.Matern52(0.7, 3.0),
        ks.kernel.Affine(1.0, 0.2, 0.0),
        ks.kernel.Wiener(0.2, 0.0, 1.0),
        ks.kernel.PiecewiseConstant(1.0, np.array([2.5])),
    ],
)
def test_cached_allocation_matches_kernel_at_absolute_times(kernel: ks.kernel.Kernel):
    fitter = RecursiveFitter(kernel)
    cache = KernelCache(kernel)
    for times in ([0.0, 1.0, 2.0], [3.0, 4.0]):
        for t in times:
            fitter.add_sample(t)
        fitter._allocate(cache)
    for j, t in enumerate(fitter.ts):
        np.testing.assert_array_equal(fitter._P_p[j], kernel.state_cov(t))
        np.testing.assert_array_equal(fitter._m_p[j], kernel.state_mean(t))
        if j:
            np.testing.assert_array_equal(fitter._A[j - 1], kernel.transition(fitter.ts[j - 1], t))
            np.testing.assert_array_equal(fitter._Q[j - 1], kernel.noise_cov(fitter.ts[j - 1], t))


def test_kernel_cache_is_shared_only_within_fit(monkeypatch: pytest.MonkeyPatch):
    kernel = ks.kernel.Exponential(1.0, 2.0)
    original = kernel.transition
    calls = []

    def transition(t1: float, t2: float):
        calls.append((t1, t2))
        return original(t1, t2)

    monkeypatch.setattr(kernel, "transition", transition)
    model = ks.BinaryModel()
    for name in "AB":
        model.add_item(name, kernel)
    for t in range(3):
        model.observe(["A"], ["B"], float(t))
    model.fit(max_iter=1)
    assert len(calls) == 1
    kernel.lscale = 4.0
    model.observe(["A"], ["B"], 3.0)
    model.fit(max_iter=1)
    assert len(calls) == 2
    fitter = model.item["A"].fitter
    assert isinstance(fitter, RecursiveFitter)
    np.testing.assert_array_equal(fitter._A[2], original(2.0, 3.0))


def test_custom_stationary_subclass_is_not_cached():
    class TimeDependentConstant(ks.kernel.Constant):
        def state_cov(self, t: float):
            return np.array([[1.0 + t]])

    kernel = TimeDependentConstant(1.0)
    cache = KernelCache(kernel)
    assert not cache.stationary
    np.testing.assert_array_equal(cache.state(1.0)[1], [[2.0]])
    np.testing.assert_array_equal(cache.state(2.0)[1], [[3.0]])
