import abc
from collections.abc import Sequence
from typing import Any, Literal

from ._native import NativeModel
from .item import Item
from .kernel import Kernel
from .observation import (
    GaussianObservation,
    LogitTieObservation,
    LogitWinObservation,
    Observation,
    PoissonObservation,
    ProbitTieObservation,
    ProbitWinObservation,
    SkellamObservation,
)


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self._item: dict[str, Item] = dict()
        self._native = NativeModel()
        self.observations: list[Observation] = list()
        self._last_method: Literal["ep", "kl"] | None = None

    @property
    def _last_method(self):
        return self._native.get_method()

    @_last_method.setter
    def _last_method(self, value):
        self._native.set_method(value)

    def _get_parameter(self):
        return self._native.parameter

    def _set_parameter(self, value):
        self._native.parameter = value

    def _outcomes(self, elems, t, kind, p=0.0, q=0.0, ternary=False):
        return tuple(self._native.outcomes([item.fitter._native for item, coeff in elems], [float(coeff) for item, coeff in elems], t, kind, p, q, ternary))

    @property
    def item(self) -> dict[str, Item]:
        return self._item

    def add_item(
        self,
        name: str,
        kernel: Kernel,
        fitter: Literal["batch", "recursive"] = "recursive",
    ) -> None:
        if name in self._item:
            raise ValueError("item '{}' already added".format(name))
        self._item[name] = Item(kernel=kernel, fitter=fitter)
        self._native.register_fitter(self._item[name].fitter._native)

    @abc.abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Add a new observation to the dataset."""

    def fit(
        self,
        method: Literal["ep", "kl"] = "ep",
        lr: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> bool:
        if method not in ("ep", "kl"):
            raise ValueError("'method' should be one of: 'ep', 'kl'")
        self._last_method = method
        return self._native.fit(method, lr, tol, max_iter, verbose)

    def _record(self, obs):
        self._native.append_observation(obs._native)
        self.observations.append(obs)

    @property
    def last_t(self):
        return self._native.last_t

    @last_t.setter
    def last_t(self, value):
        self._native.last_t = value

    def __getstate__(self):
        data = dict(self.__dict__)
        data.pop("_native")
        data["_saved_last_method"] = self._last_method
        data["_saved_parameter"] = self._native.parameter
        data["_saved_last_t"] = self.last_t
        return data

    def __setstate__(self, data):
        data = dict(data)
        last_t = data.pop("_saved_last_t")
        parameter = data.pop("_saved_parameter")
        method = data.pop("_saved_last_method")
        self.__dict__.update(data)
        self._native = NativeModel()
        self._native.parameter = parameter
        self.last_t = last_t
        self._last_method = method
        for item in self.item.values():
            self._native.register_fitter(item.fitter._native)
        for obs in self.observations:
            self._native.append_observation(obs._native)

    @abc.abstractmethod
    def probabilities(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the probability of outcomes."""

    @property
    def log_likelihood(self) -> float:
        return self._native.likelihood()

    def process_items(
        self,
        items: dict[str, Any] | list[str],
        sign: Literal[-1, +1] = +1,
    ) -> list[tuple[Item, float]]:
        if isinstance(items, dict):
            return [(self.item[k], sign * float(v)) for k, v in items.items()]
        if isinstance(items, list) or isinstance(items, tuple):
            return [(self.item[k], sign) for k in items]
        else:
            raise ValueError("items should be a list, a tuple or a dict")

    def plot_scores(
        self,
        items: Sequence[str],
        resolution: float | None = None,
        figsize: float | None = None,
        timestamps: bool = False,
    ) -> Any:
        from .plotting import plot_scores

        return plot_scores(self, items, resolution, figsize, timestamps)


class BinaryModel(Model):
    def __init__(self, obs_type: Literal["probit", "logit"] = "probit"):
        super().__init__()
        if obs_type == "probit":
            self._win_obs = ProbitWinObservation
        elif obs_type == "logit":
            self._win_obs = LogitWinObservation
        else:
            raise ValueError("unknown observation type: '{}'".format(obs_type))

    def observe(
        self,
        winners: dict[str, Any] | list[str],
        losers: dict[str, Any] | list[str],
        t: float,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        elems = self.process_items(winners, sign=+1) + self.process_items(losers, sign=-1)
        obs = self._win_obs(elems, t=t)
        self._record(obs)
        self.last_t = t

    def probabilities(
        self,
        team1: dict[str, Any] | list[str],
        team2: dict[str, Any] | list[str],
        t: float,
    ) -> tuple[float, float]:
        elems = self.process_items(team1, sign=+1) + self.process_items(team2, sign=-1)
        return self._outcomes(elems, t, self._win_obs._kind)


class TernaryModel(Model):
    margin = property(Model._get_parameter, Model._set_parameter)

    def __init__(self, margin: float = 0.1, obs_type: Literal["probit", "logit"] = "probit"):
        super().__init__()
        if obs_type == "probit":
            self._win_obs = ProbitWinObservation
            self._tie_obs = ProbitTieObservation
        elif obs_type == "logit":
            self._win_obs = LogitWinObservation
            self._tie_obs = LogitTieObservation
        else:
            raise ValueError("unknown observation type: '{}'".format(obs_type))
        self.margin = margin

    def observe(
        self,
        winners: dict[str, Any] | list[str],
        losers: dict[str, Any] | list[str],
        t: float,
        tie: bool = False,
        margin: float | None = None,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        if margin is None:
            margin = self.margin
        elems = self.process_items(winners, sign=+1) + self.process_items(losers, sign=-1)
        if tie:
            obs = self._tie_obs(elems, t=t, margin=margin)
        else:
            obs = self._win_obs(elems, t=t, margin=margin)
        self._record(obs)
        self.last_t = t

    def probabilities(
        self,
        team1: dict[str, Any] | list[str],
        team2: dict[str, Any] | list[str],
        t: float,
        margin: float | None = None,
    ) -> tuple[float, float, float]:
        if margin is None:
            margin = self.margin
        elems = self.process_items(team1, sign=+1) + self.process_items(team2, sign=-1)
        return self._outcomes(elems, t, self._win_obs._kind, margin, ternary=True)


class DifferenceModel(Model):
    var = property(Model._get_parameter, Model._set_parameter)

    def __init__(self, var: float = 1.0):
        super().__init__()
        self.var = var

    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        diff: float,
        var: float | None = None,
        t: float = 0.0,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        if var is None:
            var = self.var
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = GaussianObservation(items, diff, var, t=t)
        self._record(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        threshold: float = 0.0,
        var: float | None = None,
        t: float = 0.0,
    ) -> tuple[float, float]:
        if var is None:
            var = self.var
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        return self._outcomes(items, t, GaussianObservation._kind, threshold, var)


class CountModel(Model):
    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        count: int,
        t: float = 0.0,
    ) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = PoissonObservation(items, count, t=t)
        self._record(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        t: float = 0.0,
    ) -> tuple[float, ...]:
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        return tuple(self._native.count_probabilities([item.fitter._native for item, coeff in items], [float(coeff) for item, coeff in items], t, False, 0.0))


class CountDiffModel(Model):
    _base_rate = property(Model._get_parameter, Model._set_parameter)

    def __init__(self, base_rate: float = 0.0):
        super().__init__()
        self._base_rate = base_rate

    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        diff: int,
        t: float = 0.0,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = SkellamObservation(items, diff, self._base_rate, t=t)
        self._record(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        t: float = 0.0,
    ) -> tuple[float, ...]:
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        return tuple(self._native.count_probabilities([item.fitter._native for item, coeff in items], [float(coeff) for item, coeff in items], t, True, self._base_rate))
