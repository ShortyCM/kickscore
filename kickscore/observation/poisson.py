from .._native import loglike, moments, probability
from .observation import Observation, scalar


def _ll_poisson(x, count):
    return loglike(5, x, count, 0.0)


_ll_poisson._native_kind = 5
_ll_poisson.match_moments = lambda mean, var, count: moments(5, mean, var, count, 0.0, False)
_ll_poisson.cvi_expectations = lambda mean, var, count: moments(5, mean, var, count, 0.0, True)


def _ll_skellam(x, diff, base_rate):
    return loglike(6, x, diff, base_rate)


_ll_skellam._native_kind = 6
_ll_skellam.match_moments = lambda mean, var, diff, base_rate: moments(6, mean, var, diff, base_rate, False)
_ll_skellam.cvi_expectations = lambda mean, var, diff, base_rate: moments(6, mean, var, diff, base_rate, True)


class PoissonObservation(Observation):
    _kind = 5
    _count = scalar("p")

    def __init__(self, items, count, t):
        self._initialize(items, t, count)

    @staticmethod
    def probability(items, count, t):
        m, v = Observation.f_params(items, t)
        return probability(5, m, v, count, 0.0)


class SkellamObservation(Observation):
    _kind = 6
    _diff = scalar("p")
    _base_rate = scalar("q")

    def __init__(self, items, diff, base_rate, t):
        self._initialize(items, t, diff, base_rate)

    @staticmethod
    def probability(items, diff, base_rate, t):
        m, v = Observation.f_params(items, t)
        return probability(6, m, v, diff, base_rate)
