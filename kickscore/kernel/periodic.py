from .._native import NativeKernel
from .kernel import Kernel, parameter


class PeriodicExponential(Kernel):
    _kind = "periodic"
    _nparams = 3
    var = parameter(0)
    lscale = parameter(1)
    period = parameter(2)

    def __init__(self, var, lscale, period):
        self._native = NativeKernel(self._kind, [var, lscale, period], [], [])
