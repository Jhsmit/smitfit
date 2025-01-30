from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from smitfit.fitresult import FitResult
from smitfit.loss import Loss, SELoss
from smitfit.parameter import Parameters, pack, unpack
from smitfit.utils import flat_concat


class Minimize:  # = currently only scipy minimize
    def __init__(self, loss: Loss, parameters: Parameters, xdata: dict[str, np.ndarray]):
        self.loss = loss
        self.parameters = parameters
        self.xdata = xdata
        self.shapes = {p.name: p.shape for p in self.parameters}

    def func(self, x: np.ndarray):
        parameters = unpack(x, self.shapes)
        return self.loss(**parameters, **self.xdata)

    @property
    def free_parameters(self):
        return [p for p in self.parameters if not p.fixed]

    @property
    def fixed_parameters(self):
        return [p for p in self.parameters if p.fixed]

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        bounds = []
        for p in self.free_parameters:
            size = np.prod(p.shape, dtype=int)
            bounds += [p.bounds] * size

        if all((None, None) == b for b in bounds):
            return None
        else:
            return bounds

    def fit(self):
        x = pack(self.parameters.free.guess.values())
        result = minimize(self.func, x, bounds=self.get_bounds())
        fit_parameters = unpack(result.x, self.shapes)

        gof_qualifiers = {
            "loss": result["fun"],
        }

        std_error = {}
        if hasattr(self.loss, "y_data"):
            y_data = self.loss.y_data
            ans = self.loss.model(**self.xdata, **fit_parameters)
            f = flat_concat({k: ans[k] for k in y_data})
            y = flat_concat(y_data)

            gof_qualifiers["r_squared"] = 1 - np.sum((y - f) ** 2) / np.sum((y - np.mean(y)) ** 2)

            # move this to a function somewhere
            if isinstance(self.loss, SELoss):
                residuals = y - f

                N = len(y)
                P = len(fit_parameters)
                s_squared = np.sum(residuals**2) / (N - P)

                hess_inv = result.hess_inv  # or calculate one with numdiff
                cov_mat = s_squared * hess_inv * 2
                std_error_arr = np.sqrt(np.diag(cov_mat))
                std_error = unpack(std_error_arr, self.shapes)

        return FitResult(
            fit_parameters=fit_parameters,
            gof_qualifiers=gof_qualifiers,
            errors=std_error,
            fixed_parameters=self.parameters.fixed.guess,
            guess=self.parameters.free.guess,
            base_result=result,
        )
