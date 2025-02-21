import re
from fnmatch import fnmatch
from typing import Iterable

import sympy as sp

from smitfit.expr import as_expr
from smitfit.parameter import Parameter, Parameters
from smitfit.typing import Numerical


class Function:
    def __init__(self, func: sp.Expr | dict[sp.Symbol, sp.Expr] | str) -> None:
        if isinstance(func, dict):
            assert len(func) == 1
            self.y = list(func.keys())[0]
            self.expr = as_expr(list(func.values())[0])
        elif isinstance(func, str):
            eq = sp.parse_expr(func, evaluate=False)
            if isinstance(eq, sp.Expr):
                self.y = sp.Symbol("y")
                self.expr = as_expr(eq)
            elif isinstance(eq, sp.Equality):
                assert isinstance(eq.lhs, sp.Symbol)
                self.y = eq.lhs
                self.expr = as_expr(eq.rhs)
            else:
                raise ValueError("Invalid string expression")
        elif isinstance(func, sp.Expr):
            self.y = sp.Symbol("y")
            self.expr = as_expr(func)

        print(self.expr)

    def __call__(self, **kwargs):
        return self.expr(**kwargs)  # type: ignore

    @property
    def x_symbols(self) -> set[sp.Symbol]:
        return self.expr.symbols

    @property
    def y_symbols(self) -> set[sp.Symbol]:
        return set([self.y])

    # #TODO copy/paste code with Model -> make function
    def define_parameters(
        self, parameters: dict[str, Numerical] | Iterable[str] | str = "*"
    ) -> Parameters:
        symbols = {s.name: s for s in self.x_symbols}
        if parameters == "*":
            params = [Parameter(symbol) for symbol in self.x_symbols]
        elif isinstance(parameters, str):
            if "*" in parameters:  # fnmatch
                params = [
                    Parameter(symbol)
                    for symbol in self.x_symbols
                    if fnmatch(symbol.name, parameters)
                ]
            else:
                # split by comma, whiteplace, etc
                params = [Parameter(symbols[k.strip()]) for k in re.split(r"[,;\s]+", parameters)]

        elif isinstance(parameters, dict):
            params = [Parameter(symbols[k], guess=v) for k, v in parameters.items()]
        elif isinstance(parameters, Iterable):
            params = [Parameter(symbols[k]) for k in parameters]
        else:
            raise TypeError("Invalid type")

        return Parameters(params)
