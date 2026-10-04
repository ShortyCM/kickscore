import abc

import numpy as np

from .._arrays import array_property
from .._native import NativeObservation, f_params


def scalar(name):
    return property(lambda self: self._native.get_scalar(name), lambda self, value: self._native.set_scalar(name, float(value)))


class Observation(metaclass=abc.ABCMeta):
    _kind = 255
    t = scalar("t")
    _logpart = scalar("_logpart")
    _exp_ll = scalar("_exp_ll")
    _coeffs = array_property("_coeffs")
    _indices = array_property("_indices", int)
    _ns_cav = array_property("_ns_cav")
    _xs_cav = array_property("_xs_cav")

    @staticmethod
    @abc.abstractmethod
    def probability(*args, **kwargs):
        pass

    def __init__(self, elems, t, p=0.0, q=0.0):
        self._items = np.asarray([item for item, coeff in elems], dtype=object)
        self._M = len(elems)
        self._native = NativeObservation([item.fitter._native for item, coeff in elems], [float(coeff) for item, coeff in elems], self._kind, p, q, t)

    def match_moments(self, mean_cav, var_cav):
        return self._native.moments(mean_cav, var_cav, False)

    def cvi_expectations(self, mean, var):
        return self._native.moments(mean, var, True)

    def ep_update(self, lr=1.0):
        return self._native.update(False, lr)

    def kl_update(self, lr=0.3):
        return self._native.update(True, lr)

    @property
    def ep_log_likelihood_contrib(self):
        return self._native.likelihood(False)

    @property
    def kl_log_likelihood_contrib(self):
        return self._native.likelihood(True)

    @staticmethod
    def f_params(elems, t):
        return f_params([item.fitter._native for item, coeff in elems], [float(coeff) for item, coeff in elems], t)

    def __getstate__(self):
        data = dict(self.__dict__)
        data.pop("_native")
        data["_state"] = ([self._native.get_scalar(name) for name in ("p", "q", "t", "_logpart", "_exp_ll")], [self._native.get_array(name) for name in ("_coeffs", "_indices", "_ns_cav", "_xs_cav")])
        return data

    def __setstate__(self, data):
        data = dict(data)
        if "_state" in data:
            values, arrays = data.pop("_state")
        else:
            p = data.pop("_margin", data.pop("_diff", data.pop("_count", 0.0)))
            q = data.pop("_var", data.pop("_base_rate", 0.0))
            values = (p, q, data.pop("t"), data.pop("_logpart"), data.pop("_exp_ll"))
            arrays = [np.asarray(data.pop(name)).tolist() for name in ("_coeffs", "_indices", "_ns_cav", "_xs_cav")]
        self.__dict__.update(data)
        p, q, t, logpart, exp_ll = values
        coeffs, indices, nc, xc = arrays
        self._native = NativeObservation.restore([item.fitter._native for item in self._items], coeffs, [int(i) for i in indices], self._kind, p, q, t, nc, xc, logpart, exp_ll)
