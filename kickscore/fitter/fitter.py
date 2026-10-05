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
        state = data.pop("_state")
        self.__dict__.update(data)
        self._native = NativeFitter(self.kernel._native, self._batch)
        self._native.restore(state, self.kernel._native)
