from numpy.typing import NDArray

from ..kernel import Constant, Exponential, Kernel, Matern32, Matern52
from ..kernel.kernel import Add


def _is_stationary(kernel: Kernel) -> bool:
    if type(kernel) is Add:
        return all(_is_stationary(part) for part in kernel.parts)
    return type(kernel) in (Constant, Exponential, Matern32, Matern52)


class KernelCache:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.stationary = _is_stationary(kernel)
        self.prior: tuple[NDArray, NDArray] | None = None
        self.steps: dict[float, tuple[NDArray, NDArray]] = {}

    def state(self, t: float) -> tuple[NDArray, NDArray]:
        if not self.stationary:
            return self.kernel.state_mean(t), self.kernel.state_cov(t)
        if self.prior is None:
            self.prior = self.kernel.state_mean(t), self.kernel.state_cov(t)
        return self.prior

    def step(self, t1: float, t2: float) -> tuple[NDArray, NDArray]:
        if not self.stationary:
            return self.kernel.transition(t1, t2), self.kernel.noise_cov(t1, t2)
        delta = t2 - t1
        if delta not in self.steps:
            if len(self.steps) >= 1024:
                self.steps.clear()
            self.steps[delta] = self.kernel.transition(t1, t2), self.kernel.noise_cov(t1, t2)
        return self.steps[delta]
