from .._native import NativeKernel
from .kernel import Kernel, parameter


class Exponential(Kernel):
    _kind = "exponential"
    _nparams = 2
    var = parameter(0)
    lscale = parameter(1)

    def __init__(self, var, lscale):
        self._native = NativeKernel(self._kind, [var, lscale], [], [])

    @property
    def stationary_mean(self):
        return self.state_mean(0.0)

    @property
    def stationary_cov(self):
        return self.state_cov(0.0)
