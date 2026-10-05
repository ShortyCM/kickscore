from math import pi, sqrt

from .._native import iv, log_factorial, logphi, logsumexp, logsumexp2, moments, normcdf, normpdf

SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)


def cvi_expectations(ll_fct):
    if not hasattr(ll_fct, "_native_kind"):
        raise NotImplementedError("custom Python likelihood callbacks are not supported by the Rust core")
    kind = ll_fct._native_kind
    ll_fct.cvi_expectations = lambda mean, var, p=0.0, q=0.0: moments(kind, mean, var, p, q, True)
    return ll_fct


def match_moments(ll_fct):
    if not hasattr(ll_fct, "_native_kind"):
        raise NotImplementedError("custom Python likelihood callbacks are not supported by the Rust core")
    kind = ll_fct._native_kind
    ll_fct.match_moments = lambda mean, var, p=0.0, q=0.0: moments(kind, mean, var, p, q, False)
    return ll_fct
