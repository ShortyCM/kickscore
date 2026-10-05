from .._native import moments, probability
from .observation import Observation, scalar


def _mm_gaussian(mean_cav, var_cav, diff, var_obs):
    return moments(4, mean_cav, var_cav, diff, var_obs, False)


class GaussianObservation(Observation):
    _kind = 4
    _diff = scalar("p")
    _var = scalar("q")

    def __init__(self, items, diff, var, t):
        self._initialize(items, t, diff, var)

    @staticmethod
    def probability(items, threshold, var, t):
        m, v = Observation.f_params(items, t)
        return probability(4, m, v, threshold, var)
