import numpy as np

from .fitter import Fitter


class RecursiveFitter(Fitter):
    _batch = False

    def __init__(self, kernel):
        super().__init__(kernel)

    def _allocate(self, cache):
        self.allocate()

    def __getattr__(self, name):
        if name in ("_A", "_Q", "_m_p", "_P_p", "_m_f", "_P_f", "_m_s", "_P_s"):
            data = np.asarray(self._native.matrices(name))
            order = self.kernel.order
            if name.startswith("_m"):
                return data.reshape(-1, order)
            return data.reshape(-1, order, order)
        if name == "_h":
            return self.kernel.measurement_vector
        if name == "_I":
            return np.eye(self.kernel.order)
        raise AttributeError(name)
