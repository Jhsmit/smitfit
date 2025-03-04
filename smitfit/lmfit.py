import lmfit as lm
import numpy as np

from smitfit.loss import SELoss
from smitfit.model import Model
from smitfit.parameter import Parameters
from smitfit.result import Result
from smitfit.utils import flat_concat


class Minimize:
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
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
