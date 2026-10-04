import numpy as np


def array_property(name, dtype=float):
    def get(self):
        return np.asarray(self._native.buffer(name))

    def set_(self, value):
        self._native.set_array(name, np.asarray(value, dtype=dtype).tolist())

    return property(get, set_)


class PendingSamples:
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return len(self.owner.get_array("ts_new"))

    def __iter__(self):
        return iter(self.owner.get_array("ts_new"))

    def __getitem__(self, index):
        return self.owner.get_array("ts_new")[index]

    def __setitem__(self, index, value):
        values = self.owner.get_array("ts_new")
        values[index] = value
        self.owner.set_array("ts_new", values)

    def __delitem__(self, index):
        values = self.owner.get_array("ts_new")
        del values[index]
        self.owner.set_array("ts_new", values)

    def append(self, value):
        self.owner.append_pending(value)

    def extend(self, values):
        for value in values:
            self.append(value)

    def insert(self, index, value):
        values = self.owner.get_array("ts_new")
        values.insert(index, value)
        self.owner.set_array("ts_new", values)

    def clear(self):
        self.owner.set_array("ts_new", [])

    def pop(self, index=-1):
        values = self.owner.get_array("ts_new")
        value = values.pop(index)
        self.owner.set_array("ts_new", values)
        return value

    def __eq__(self, other):
        return self.owner.get_array("ts_new") == other

    def __repr__(self):
        return repr(self.owner.get_array("ts_new"))
