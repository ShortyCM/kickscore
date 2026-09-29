from math import isfinite
from typing import Literal

import numba
import numpy as np
from numpy.typing import NDArray

from ..fitter import Fitter
from .observation import Observation
from .ordinal import (
    LogitTieObservation,
    LogitWinObservation,
    ProbitTieObservation,
    ProbitWinObservation,
    _ll_logit_tie,
    _ll_logit_win,
    _ll_probit_tie,
    _ll_probit_win,
    _mm_logit_win,
    _mm_probit_tie,
    _mm_probit_win,
)

_CVI_PROBIT_WIN = getattr(_ll_probit_win, "cvi_expectations")
_CVI_LOGIT_WIN = getattr(_ll_logit_win, "cvi_expectations")
_CVI_PROBIT_TIE = getattr(_ll_probit_tie, "cvi_expectations")
_CVI_LOGIT_TIE = getattr(_ll_logit_tie, "cvi_expectations")
_MM_LOGIT_TIE = getattr(_ll_logit_tie, "match_moments")
_KINDS: dict[type[Observation], int] = {
    ProbitWinObservation: 0,
    LogitWinObservation: 1,
    ProbitTieObservation: 2,
    LogitTieObservation: 3,
}


@numba.njit
def _moments(
    kind: int, mean: float, var: float, margin: float, kl: bool
) -> tuple[float, float, float]:
    if kl:
        if kind == 0:
            return _CVI_PROBIT_WIN(mean, var, margin)
        if kind == 1:
            return _CVI_LOGIT_WIN(mean, var, margin)
        if kind == 2:
            return _CVI_PROBIT_TIE(mean, var, margin)
        return _CVI_LOGIT_TIE(mean, var, margin)
    if kind == 0:
        return _mm_probit_win(mean - margin, var)
    if kind == 1:
        return _mm_logit_win(mean - margin, var)
    if kind == 2:
        return _mm_probit_tie(mean, var, margin)
    return _MM_LOGIT_TIE(mean, var, margin)


@numba.njit
def _update(
    offsets: NDArray,
    indices: NDArray,
    coeffs: NDArray,
    kinds: NDArray,
    margins: NDArray,
    ms: NDArray,
    vs: NDArray,
    ns: NDArray,
    xs: NDArray,
    ns_cav: NDArray,
    xs_cav: NDArray,
    values: NDArray,
    lr: float,
    kl: bool,
) -> float:
    max_diff = 0.0
    for j in range(len(kinds)):
        mean, var = 0.0, 0.0
        for i in range(offsets[j], offsets[j + 1]):
            idx, coeff = indices[i], coeffs[i]
            if kl:
                mean += coeff * ms[idx]
                var += coeff * coeff * vs[idx]
            else:
                x_tot = 1.0 / vs[idx]
                n_tot = x_tot * ms[idx]
                x_cav = x_tot - xs[idx]
                n_cav = n_tot - ns[idx]
                if not isfinite(x_cav) or x_cav <= 0.0:
                    raise FloatingPointError("invalid EP cavity precision")
                xs_cav[i], ns_cav[i] = x_cav, n_cav
                mean += coeff * n_cav / x_cav
                var += coeff * coeff / x_cav
        value, first, second = _moments(kinds[j], mean, var, margins[j], kl)
        if not (isfinite(value) and isfinite(first) and isfinite(second)):
            raise FloatingPointError("non-finite observation update")
        for i in range(offsets[j], offsets[j + 1]):
            idx, coeff = indices[i], coeffs[i]
            if kl:
                x = -2.0 * coeff * coeff * second
                n = coeff * (first - 2 * ms[idx] * coeff * second)
            else:
                x_cav, n_cav = xs_cav[i], ns_cav[i]
                denom = 1 + coeff * coeff * second / x_cav
                if not isfinite(denom) or denom <= 0.0:
                    raise FloatingPointError("invalid EP posterior variance")
                x = -coeff * coeff * second / denom
                n = coeff * (first - coeff * (n_cav / x_cav) * second) / denom
            xs[idx] = (1 - lr) * xs[idx] + lr * x
            ns[idx] = (1 - lr) * ns[idx] + lr * n
            if not (isfinite(xs[idx]) and isfinite(ns[idx])):
                raise FloatingPointError("non-finite pseudo-observation")
        diff = abs(values[j] - value)
        if not isfinite(diff):
            raise FloatingPointError("non-finite convergence difference")
        max_diff = max(max_diff, diff)
        values[j] = value
    return max_diff


class ObservationBatch:
    @classmethod
    def create(
        cls, observations: list[Observation], fitters: list[Fitter]
    ) -> "ObservationBatch | None":
        overrides = ("ep_update", "kl_update", "match_moments", "cvi_expectations")
        if not observations or any(
            type(obs) not in _KINDS or any(name in vars(obs) for name in overrides)
            for obs in observations
        ):
            return None
        return cls(observations, fitters)

    def __init__(self, observations: list[Observation], fitters: list[Fitter]):
        self.observations = observations
        self.fitters = fitters
        self.fitter_offsets = np.cumsum([0] + [len(f.ts) for f in fitters])
        starts = {id(f): start for f, start in zip(fitters, self.fitter_offsets)}
        self.offsets = np.cumsum([0] + [obs._M for obs in observations])
        self.indices = np.concatenate(
            [
                np.array([starts[id(item.fitter)] for item in obs._items]) + obs._indices
                for obs in observations
            ]
        )
        self.coeffs = np.concatenate([obs._coeffs for obs in observations])
        self.kinds = np.array([_KINDS[type(obs)] for obs in observations])
        self.margins = np.array([getattr(obs, "_margin") for obs in observations], dtype=float)
        self.ns_cav = np.concatenate([obs._ns_cav for obs in observations])
        self.xs_cav = np.concatenate([obs._xs_cav for obs in observations])
        self.logparts = np.array([obs._logpart for obs in observations], dtype=float)
        self.exp_lls = np.array([obs._exp_ll for obs in observations], dtype=float)
        self.ms = np.empty(self.fitter_offsets[-1])
        self.vs = np.empty_like(self.ms)
        self.ns = np.empty_like(self.ms)
        self.xs = np.empty_like(self.ms)

    def update(self, method: Literal["ep", "kl"], lr: float) -> float:
        for f, start, end in zip(self.fitters, self.fitter_offsets, self.fitter_offsets[1:]):
            self.ms[start:end], self.vs[start:end] = f.ms, f.vs
            self.ns[start:end], self.xs[start:end] = f.ns, f.xs
        values = self.logparts if method == "ep" else self.exp_lls
        diff = _update(
            self.offsets,
            self.indices,
            self.coeffs,
            self.kinds,
            self.margins,
            self.ms,
            self.vs,
            self.ns,
            self.xs,
            self.ns_cav,
            self.xs_cav,
            values,
            lr,
            method == "kl",
        )
        for f, start, end in zip(self.fitters, self.fitter_offsets, self.fitter_offsets[1:]):
            f.ns[:], f.xs[:] = self.ns[start:end], self.xs[start:end]
        for j, obs in enumerate(self.observations):
            if method == "ep":
                obs._logpart = values[j]
                start, end = self.offsets[j : j + 2]
                obs._ns_cav[:] = self.ns_cav[start:end]
                obs._xs_cav[:] = self.xs_cav[start:end]
            else:
                obs._exp_ll = values[j]
        return diff
