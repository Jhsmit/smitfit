import numpy as np
from scipy.optimize import minimize, LbfgsInvHessProduct
from smitfit.result import Result
from smitfit.loss import Loss, SELoss
from smitfit.parameter import Parameters, pack, unpack
from smitfit.utils import flat_concat
import lmfit as lm


class Minimize:  # = currently only scipy minimize
    def __init__(
        self,
        model,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
        parameters: Parameters,
    ):
        self.loss = SELoss(model, ydata)
        self.parameters = parameters
        self.xdata = xdata

    def fit(self):
        lm_params = lm.Parameters()
        for par in self.parameters:
            lm_params.add(
                par.symbol.name,
                value=par.guess,
                min=par.bounds[0],
                max=par.bounds[1],
                vary=not par.fixed,
            )

        def residual(param):
            return flat_concat(self.loss.residuals(**self.xdata, **param.valuesdict()))

        result = lm.minimize(residual, lm_params)
        fit_parameters = {k.name: result.params[k.name].value for k in self.parameters.free}

        # redchi, aic, bic
        gof_qualifiers = {"chisqr": result.chisqr}
        errors = {k.name: result.uvars[k.name].std_dev for k in self.parameters.free}

        # fixed_parameters = {k.name: result.params[k.name].value for k in self.parameters.fixed}
        # check identical with self.parameters.fixed.guess

        return Result(
            fit_parameters=fit_parameters,
            gof_qualifiers=gof_qualifiers,
            errors=errors,
            fixed_parameters=self.parameters.fixed.guess,
            guess=self.parameters.free.guess,
            base_result=result,
        )
