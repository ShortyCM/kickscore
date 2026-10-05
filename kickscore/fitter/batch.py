from .fitter import Fitter


class BatchFitter(Fitter):
    _batch = True

    def __init__(self, kernel):
        super().__init__(kernel)
