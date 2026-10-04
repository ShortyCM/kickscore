import abc

import numpy as np

from .._arrays import PendingSamples, array_property
from .._native import NativeFitter


class Fitter(metaclass=abc.ABCMeta):
    ts = array_property("ts")
    ms = array_property("ms")
    vs = array_property("vs")
    ns = array_property("ns")
    xs = array_property("xs")

    @abc.abstractmethod
    def __init__(self, kernel):
        kernel._check_native_methods()
        self.kernel = kernel
        self._native = NativeFitter(kernel._native, self._batch)

    @property
    def ts_new(self):
        return PendingSamples(self._native)

    @ts_new.setter
    def ts_new(self, values):
        self._native.set_array("ts_new", list(values))

    @property
    def is_fitted(self):
        return self._native.is_fitted()

    @is_fitted.setter
    def is_fitted(self, value):
        self._native.set_fitted(value)

    @property
    def is_allocated(self):
        return self._native.is_allocated()

    def add_sample(self, t):
        return self._native.add_sample(t)

    def allocate(self):
        self._native.allocate()

    def fit(self):
        self._native.fit()

    @property
    def posterior(self):
        self._native.ready()
        return self.ts, self.ms, self.vs

    @property
    def ep_log_likelihood_contrib(self):
        return self._native.likelihood(False)

    @property
    def kl_log_likelihood_contrib(self):
        return self._native.likelihood(True)

    def predict(self, ts):
        mean, var = self._native.predict(np.asarray(ts, dtype=float).tolist())
        return np.asarray(mean), np.asarray(var)

    def __getstate__(self):
        data = dict(self.__dict__)
        data.pop("_native")
        data["_state"] = bytes(self._native.dump())
        return data

    def __setstate__(self, data):
        data = dict(data)
        state = data.pop("_state", None)
        self.kernel = data.pop("kernel")
        self._native = NativeFitter(self.kernel._native, self._batch)
        if state is not None:
            self.__dict__.update(data)
            self._native.restore(state, self.kernel._native)
            return
        ts = np.asarray(data.pop("ts"), dtype=float)
        pending = data.pop("ts_new")
        fitted = data.pop("is_fitted")
        for t in ts:
            self._native.add_sample(float(t))
        self._native.allocate()
        for name in ("ms", "vs", "ns", "xs"):
            self._native.set_array(name, np.asarray(data.pop(name), dtype=float).tolist())
        self._native.set_array("ts_new", list(pending))
        if self._batch:
            for name in ("_k_mat", "_cov", "_b_cholesky", "_woodbury_inv", "_woodbury_vec"):
                value = data.pop(name, None)
                if value is not None and np.asarray(value).size:
                    array = np.asarray(value, dtype=float)
                    self._native.set_matrix(name, array.reshape(array.shape[0], -1).tolist())
        else:
            for name in ("_A", "_Q", "_m_p", "_P_p", "_m_f", "_P_f", "_m_s", "_P_s"):
                values = np.asarray(data.pop(name), dtype=float)
                if name in ("_A", "_Q"):
                    values = values[:max(0, len(ts) - 1)]
                self._native.set_history(name, values.reshape(len(values), -1).tolist() if len(values) else [])
            data.pop("_h", None)
            data.pop("_I", None)
        self.__dict__.update(data)
        self._native.set_fitted(fitted)
