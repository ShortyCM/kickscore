import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class ArrayProxy(NDArrayOperatorsMixin):
    def __init__(self, owner, name, dtype=float):
        self.owner = owner
        self.name = name
        self.dtype = np.dtype(dtype)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.owner.get_array(self.name), dtype=dtype or self.dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outputs = kwargs.pop("out", None)
        args = [np.asarray(x) if isinstance(x, ArrayProxy) else x for x in inputs]
        result = getattr(ufunc, method)(*args, **kwargs)
        if outputs is not None:
            values = result if isinstance(result, tuple) else (result,)
            for target, value in zip(outputs, values):
                target[:] = value
            return outputs[0] if len(outputs) == 1 else outputs
        return result

    def __len__(self):
        return len(self.owner.get_array(self.name))

    def __iter__(self):
        return iter(np.asarray(self))

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __setitem__(self, key, value):
        data = np.asarray(self)
        data[key] = value
        self.owner.set_array(self.name, data.tolist())

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __repr__(self):
        return repr(np.asarray(self))


def array_property(name, dtype=float):
    def get(self):
        return ArrayProxy(self._native, name, dtype)

    def set_(self, value):
        self._native.set_array(name, np.asarray(value, dtype=dtype).tolist())

    return property(get, set_)
