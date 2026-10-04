from .._native import loglike, moments, probability
from .observation import Observation, scalar


def _mm_probit_win(mean_cav, cov_cav):
    return moments(0, mean_cav, cov_cav, 0.0, 0.0, False)


def _mm_probit_tie(mean_cav, cov_cav, margin):
    return moments(2, mean_cav, cov_cav, margin, 0.0, False)


def _mm_logit_win(mean_cav, cov_cav):
    return moments(1, mean_cav, cov_cav, 0.0, 0.0, False)


def _likelihood(kind):
    def ll(x, margin):
        return loglike(kind, x, margin, 0.0)
    ll._native_kind = kind
    ll.cvi_expectations = lambda mean, var, margin: moments(kind, mean, var, margin, 0.0, True)
    ll.match_moments = lambda mean, var, margin: moments(kind, mean, var, margin, 0.0, False)
    return ll


_ll_probit_win = _likelihood(0)
_ll_logit_win = _likelihood(1)
_ll_probit_tie = _likelihood(2)
_ll_logit_tie = _likelihood(3)


class ProbitWinObservation(Observation):
    _kind = 0
    _margin = scalar("p")

    def __init__(self, elems, t, margin=0):
        super().__init__(elems, t, margin)

    @staticmethod
    def probability(elems, t, margin=0):
        m, v = Observation.f_params(elems, t)
        return probability(0, m, v, margin, 0.0)


class LogitWinObservation(Observation):
    _kind = 1
    _margin = scalar("p")

    def __init__(self, elems, t, margin=0):
        super().__init__(elems, t, margin)

    @staticmethod
    def probability(elems, t, margin=0):
        m, v = Observation.f_params(elems, t)
        return probability(1, m, v, margin, 0.0)


class ProbitTieObservation(Observation):
    _kind = 2
    _margin = scalar("p")

    def __init__(self, elems, t, margin):
        super().__init__(elems, t, margin)

    @staticmethod
    def probability(elems, t, margin=0):
        if margin == 0:
            return 0.0
        m, v = Observation.f_params(elems, t)
        return probability(2, m, v, margin, 0.0)


class LogitTieObservation(Observation):
    _kind = 3
    _margin = scalar("p")

    def __init__(self, elems, t, margin=0):
        super().__init__(elems, t, margin)

    @staticmethod
    def probability(elems, t, margin=0):
        m, v = Observation.f_params(elems, t)
        return probability(3, m, v, margin, 0.0)
