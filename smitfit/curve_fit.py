from smitfit.result import Result
from smitfit.function import Function
from smitfit.parameter import Parameters, pack, unpack
import numpy as np
from scipy.optimize import curve_fit


class CurveFit:
    def __init__(self, func: Function, xdata: dict, ydata: dict, parameters: Parameters):
        self.func = func
        self.xdata = xdata
        self.ydata = ydata
        self.parameters = parameters

    def f(self, xdata: np.ndarray, *args):
        unstacked_x = {k: v for k, v in zip(self.xdata, np.atleast_2d(xdata))}
        kwargs = unpack(args, self.parameters.free.shapes)
        return self.func(**unstacked_x, **kwargs, **self.parameters.fixed.guess)

    def fit(self) -> Result:
        p0 = pack(self.parameters.free.guess.values())
        ydata = self.ydata[self.func.y.name]
        xdata = np.stack(list(self.xdata.values()))

        popt, pcov, infodict, mesg, ier = curve_fit(self.f, xdata, ydata, p0=p0, full_output=True)
        base_result = dict(popt=popt, pcov=pcov, infodict=infodict, mesg=mesg, ier=ier)

        errors = unpack(np.sqrt(np.diag(pcov)), self.parameters.free.shapes)
        parameters = unpack(popt, self.parameters.free.shapes)

        result = Result(
            fit_parameters=parameters,
            gof_qualifiers={},
            errors=errors,  # type: ignore
            fixed_parameters=self.parameters.fixed.guess,
            guess=self.parameters.guess,
            base_result=base_result,
        )
        return result
