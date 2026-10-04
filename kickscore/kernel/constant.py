import numpy as np

from .._native import NativeKernel
from .kernel import Kernel, parameter


class Constant(Kernel):
    _kind = "constant"
    _nparams = 1
    var = parameter(0)

    def __init__(self, var):
        self._native = NativeKernel(self._kind, [var], [], [])


class PiecewiseConstant(Kernel):
    _kind = "piecewise"
    _nparams = 1
    var = parameter(0)

    def __init__(self, var, bounds):
        self._native = NativeKernel(self._kind, [var], np.asarray(bounds, dtype=float).tolist(), [])

    @property
    def bounds(self):
        return np.asarray(self._native.bounds())

    @bounds.setter
    def bounds(self, value):
        self._native.set_bounds(np.asarray(value, dtype=float).tolist())
