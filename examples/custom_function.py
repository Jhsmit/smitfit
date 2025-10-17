"""
Example of using custom python functions or numpy functions in smitfit models.

"""

import numpy as np
import ultraplot as uplt

from smitfit.composite_expr import FuncExpr
from smitfit.loss import SELoss
from smitfit.minimize import Minimize
from smitfit.model import Model
import smitfit.numpy_func as npf
import sympy as sp

from smitfit.symbol import Symbols

# %%

# Generate Ground-Truth data
np.random.seed(43)
gt = {"a": 0.15, "b": 2.5}

x_arr = np.linspace(0, 11, num=100)
y_arr = gt["a"] * x_arr + gt["b"]

noise = np.random.normal(0, scale=y_arr / 10.0 + 0.2)
y_arr += noise

xdata, ydata = {"x": x_arr}, {"y": y_arr}

# %%
# Define a custom function to use in the model
# args can be any object which is convertible to an Expr


def func(a, x, b):
    return a * x + b


custom_func = FuncExpr(func, *sp.symbols("a x b"))

# %%
model = Model({sp.Symbol("y"): custom_func})
parameters = model.define_parameters("a b")

# %%
loss = SELoss(model, ydata)
minimize = Minimize(loss, parameters, xdata)
result = minimize.fit()
result.parameters

# %%
# We can also use any numpy ufunc as FuncExpr
# by importing from the smitfit.numpy_func module
sym = Symbols("a x b")
func_np = npf.add(sym.a * sym.x, sym.b)  # type: ignore
y_numpy = func_np(**result.parameters, **xdata)

fig, ax = uplt.subplots()
ax.scatter(xdata["x"], ydata["y"])
ax.plot(xdata["x"], model(**result.parameters, **xdata)["y"], color="r")
ax.plot(xdata["x"], y_numpy, color="k", linestyle="--")
ax.format(xlabel="x", ylabel="y")
# %%
