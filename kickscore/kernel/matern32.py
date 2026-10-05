from .._native import NativeKernel
from .kernel import Kernel
from .kernel import parameter


class Matern32(Kernel):
    _kind = "matern32"
    _nparams = 3
    var = parameter(0)
    lscale = parameter(1)
    lambda_ = parameter(2)

    def __init__(self, var, lscale):
        self._native = NativeKernel(self._kind, [var, lscale], [], [])

    @property
    def stationary_mean(self):
        return self.state_mean(0.0)

    @property
    def stationary_cov(self):
        return self.state_cov(0.0)
