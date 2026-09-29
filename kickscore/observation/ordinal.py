from collections.abc import Sequence
from math import exp, expm1, log, log1p, sqrt  # Faster than numpy equivalents.

import numba
import numpy as np

from ..item import Item
from .observation import Observation
from .utils import SQRT2PI, cvi_expectations, logphi, logsumexp2, match_moments


@numba.jit(nopython=True)
def _mm_probit_win(mean_cav: float, cov_cav: float) -> tuple[float, float, float]:
    # Adapted from the GPML function `likErf.m`.
    z = mean_cav / sqrt(1 + cov_cav)
    logpart, val = logphi(z)
    dlogpart = val / sqrt(1 + cov_cav)  # 1st derivative w.r.t. mean.
    d2logpart = -val * (z + val) / (1 + cov_cav)
    return logpart, dlogpart, d2logpart


@cvi_expectations
@numba.jit(nopython=True)
def _ll_probit_win(x: float, margin: float) -> float:
    """Compute log-likelihood of x under the probit win model."""
    return logphi(x - margin)[0]


class ProbitWinObservation(Observation):
    def __init__(self, elems: Sequence[tuple[Item, float]], t: float, margin: float = 0):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _mm_probit_win(mean_cav - self._margin, var_cav)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_probit_win.cvi_expectations(mean, var, self._margin)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(elems: Sequence[tuple[Item, float]], t: float, margin: float = 0) -> float:
        m, v = Observation.f_params(elems, t)
        logpart, _, _ = _mm_probit_win(m - margin, v)
        return exp(logpart)


@numba.jit(nopython=True)
def _mm_probit_tie(mean_cav: float, cov_cav: float, margin: float) -> tuple[float, float, float]:
    if margin <= 0:
        raise ValueError("tie margin must be positive")
    denom = sqrt(1 + cov_cav)
    z1 = (-abs(mean_cav) + margin) / denom
    z2 = (-abs(mean_cav) - margin) / denom
    logcdf1 = logphi(z1)[0]
    logcdf2 = logphi(z2)[0]
    logpart = logcdf1 + log(-expm1(logcdf2 - logcdf1))
    v1 = exp(-0.5 * z1 * z1 - log(SQRT2PI) - logpart)
    v2 = exp(-0.5 * z2 * z2 - log(SQRT2PI) - logpart)
    dlogpart = (v1 - v2) / denom
    d2logpart = (-z1 * v1 + z2 * v2) / (1 + cov_cav) - dlogpart**2
    if mean_cav > 0:
        dlogpart = -dlogpart
    return logpart, dlogpart, d2logpart


@cvi_expectations
@numba.jit(nopython=True)
def _ll_probit_tie(x: float, margin: float) -> float:
    """Compute log-likelihood of x under the probit tie model."""
    # Stable computation log(1 - e^a) c.f.
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -abs(x)  # Stabilizes computations (likelihood is symmetric).
    z = logphi(x + margin)[0]
    a = logphi(x - margin)[0] - z
    if a > -0.693:  # ~= log 2.
        return z + log(-expm1(a))
    else:
        return z + log1p(-exp(a))


class ProbitTieObservation(Observation):
    def __init__(self, elems: Sequence[tuple[Item, float]], t: float, margin: float):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _mm_probit_tie(mean_cav, var_cav, self._margin)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_probit_tie.cvi_expectations(mean, var, self._margin)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(elems: Sequence[tuple[Item, float]], t: float, margin: float = 0) -> float:
        if margin == 0:
            return 0.0
        m, v = Observation.f_params(elems, t)
        logpart, _, _ = _mm_probit_tie(m, v, margin)
        return exp(logpart)


# Constants used in `_mm_logit_win`.
LAMBDAS = sqrt(2) * np.array([0.44, 0.41, 0.40, 0.39, 0.36])
CS = np.array(
    [
        1.146480988574439e02,
        -1.508871030070582e03,
        2.676085036831241e03,
        -1.356294962039222e03,
        7.543285642111850e01,
    ]
)


@numba.jit(nopython=True)
def _mm_logit_win(mean_cav: float, cov_cav: float) -> tuple[float, float, float]:
    exponent = -10.0 * (abs(mean_cav) - (196.0 / 200.0) * cov_cav - 4.0)
    if exponent < 500:
        lambd = 1.0 / (1.0 + exp(exponent))
        logpart2 = min(cov_cav / 2.0 - abs(mean_cav), -0.1)
        dlogpart2 = 1.0
        if mean_cav > 0:
            logpart2 = log1p(-exp(logpart2))
            dlogpart2 = 0.0
        if lambd == 1.0:
            return logpart2, dlogpart2, 0.0
    else:
        lambd, logpart2, dlogpart2 = 0.0, 0.0, 0.0
    arr1, arr2, arr3 = np.empty(5), np.empty(5), np.empty(5)
    for i, x in enumerate(LAMBDAS):
        arr1[i], arr2[i], arr3[i] = _mm_probit_win(x * mean_cav, x * x * cov_cav)
    logpart1 = logsumexp2(arr1, CS)
    weights = np.exp(arr1 - np.max(arr1))
    denom = np.dot(weights, CS)
    dlogpart1 = np.dot(weights * arr2, CS * LAMBDAS) / denom
    d2logpart1 = (np.dot(weights * (arr2 * arr2 + arr3), CS * LAMBDAS * LAMBDAS) / denom) - (
        dlogpart1 * dlogpart1
    )
    logpart = (1 - lambd) * logpart1 + lambd * logpart2
    dlogpart = (1 - lambd) * dlogpart1 + lambd * dlogpart2
    d2logpart = (1 - lambd) * d2logpart1
    return logpart, dlogpart, d2logpart


@cvi_expectations
@numba.jit(nopython=True)
def _ll_logit_win(x: float, margin: float) -> float:
    """Compute log-likelihood of x under the logit win model."""
    z = x - margin
    if z > 0:
        return -log1p(exp(-z))
    else:
        return z - log1p(exp(z))


class LogitWinObservation(Observation):
    def __init__(self, elems: Sequence[tuple[Item, float]], t: float, margin: float = 0):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _mm_logit_win(mean_cav - self._margin, var_cav)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_logit_win.cvi_expectations(mean, var, self._margin)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(elems: Sequence[tuple[Item, float]], t: float, margin: float = 0) -> float:
        m, v = Observation.f_params(elems, t)
        logpart, _, _ = _mm_logit_win(m - margin, v)
        return exp(logpart)


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_logit_tie(x: float, margin: float) -> float:
    """Compute log-likelihood of x under the logit tie model."""
    return _ll_logit_win(x, margin) + _ll_logit_win(-x, margin) + log(expm1(2 * margin))


class LogitTieObservation(Observation):
    def __init__(self, elems: Sequence[tuple[Item, float]], t: float, margin: float = 0):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _ll_logit_tie.match_moments(mean_cav, var_cav, self._margin)  # pyright: ignore[reportFunctionMemberAccess]

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_logit_tie.cvi_expectations(mean, var, self._margin)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(elems: Sequence[tuple[Item, float]], t: float, margin: float = 0) -> float:
        m, v = Observation.f_params(elems, t)
        logpart1, _, _ = _mm_logit_win(+m - margin, v)
        logpart2, _, _ = _mm_logit_win(-m - margin, v)
        return 1.0 - exp(logpart1) - exp(logpart2)
