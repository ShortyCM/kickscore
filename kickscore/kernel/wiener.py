from .._native import NativeKernel
from .kernel import Kernel, parameter


class Wiener(Kernel):
    _kind = "wiener"
    _nparams = 3
    var = parameter(0)
    t0 = parameter(1)
    var_t0 = parameter(2)

    def __init__(self, var, t0, var_t0=0.0):
        self._native = NativeKernel(self._kind, [var, t0, var_t0], [], [])
