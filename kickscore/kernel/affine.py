from .._native import NativeKernel
from .kernel import Kernel, parameter


class Affine(Kernel):
    _kind = "affine"
    _nparams = 3
    var_offset = parameter(0)
    var_slope = parameter(1)
    t0 = parameter(2)

    def __init__(self, var_offset, var_slope, t0):
        self._native = NativeKernel(self._kind, [var_offset, var_slope, t0], [], [])
