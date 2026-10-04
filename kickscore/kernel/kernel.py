import abc

import numpy as np

from .._native import NativeKernel


def parameter(index):
    return property(lambda self: self._native.get_param(index), lambda self, value: self._native.set_param(index, float(value)))


class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    def _check_native_methods(self):
        methods = ("k_mat", "k_diag", "state_cov", "state_mean", "transition", "noise_cov")
        for name in methods:
            if name in vars(self) or getattr(type(self), name) is not getattr(Kernel, name):
                raise NotImplementedError("Python kernel overrides require Python callbacks; native fitting does not execute them")
        for part in getattr(self, "parts", []):
            part._check_native_methods()

    def k_mat(self, ts1, ts2=None):
        first = np.asarray(ts1, dtype=float).tolist()
        second = first if ts2 is None else np.asarray(ts2, dtype=float).tolist()
        return np.asarray(self._native.k_mat(first, second)).reshape(len(first), len(second))

    def k_diag(self, ts):
        return np.asarray(self._native.k_diag(np.asarray(ts, dtype=float).tolist()))

    @property
    def order(self):
        return self._native.order()

    def state_mean(self, t):
        return np.asarray(self._native.mean())

    def state_cov(self, t):
        return np.asarray(self._native.matrix("state", t, t))

    @property
    def measurement_vector(self):
        return np.asarray(self._native.h())

    @property
    def feedback(self):
        return np.asarray(self._native.matrix("feedback", 0.0, 0.0))

    @property
    def noise_effect(self):
        return np.asarray(self._native.matrix("effect", 0.0, 0.0))

    @property
    def noise_density(self):
        return np.asarray(self._native.matrix("density", 0.0, 0.0))

    def transition(self, t1, t2):
        return np.asarray(self._native.matrix("transition", t1, t2))

    def noise_cov(self, t1, t2):
        return np.asarray(self._native.matrix("noise", t1, t2))

    def __add__(self, other):
        return Add(self, other)

    @staticmethod
    def distances(ts1, ts2):
        from .._native import distances
        return np.asarray(distances(np.asarray(ts1, dtype=float).tolist(), np.asarray(ts2, dtype=float).tolist())).reshape(len(ts1), len(ts2))

    def simulate(self, ts):
        return np.asarray(self._native.simulate(np.asarray(ts, dtype=float).tolist()))

    def __getstate__(self):
        data = dict(self.__dict__)
        data.pop("_native")
        data["_params"] = [self._native.get_param(i) for i in range(self._nparams)]
        data["_bounds"] = self._native.bounds()
        return data

    def __setstate__(self, data):
        fields = {
            "constant": ("var",), "piecewise": ("var",),
            "exponential": ("var", "lscale"),
            "matern32": ("var", "lscale", "lambda_"),
            "matern52": ("var", "lscale", "lambda_"),
            "affine": ("var_offset", "var_slope", "t0"),
            "wiener": ("var", "t0", "var_t0"),
            "periodic": ("var", "lscale", "period"), "add": (),
        }
        if "_params" in data:
            params = data.pop("_params")
            bounds = data.pop("_bounds")
        else:
            params = [data.pop(name) for name in fields[self._kind]]
            bounds = np.asarray(data.pop("bounds", []), dtype=float).tolist()
        self.__dict__.update(data)
        self._native = NativeKernel(self._kind, params, bounds, [k._native for k in getattr(self, "parts", [])])


class Add(Kernel):
    _kind = "add"
    _nparams = 0

    def __init__(self, first, second):
        self.parts = []
        for kernel in (first, second):
            self.parts.extend(kernel.parts if isinstance(kernel, Add) else [kernel])
        self._native = NativeKernel("add", [], [], [k._native for k in self.parts])
