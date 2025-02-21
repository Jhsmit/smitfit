from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Iterable

import sympy as sp
from toposort import toposort

from smitfit.expr import Expr, as_expr
from smitfit.parameter import Parameter, Parameters
from smitfit.typing import Numerical


def parse_model_str(model: Iterable[str]) -> dict[sp.Symbol, sp.Expr]:
    model_dict = {}
    for s in model:
        eq = sp.parse_expr(s, evaluate=False)
        if not isinstance(eq.lhs, sp.Symbol):
            raise ValueError("lhs must be a symbol")
        if not isinstance(eq.rhs, sp.Expr):
            raise ValueError("rhs must be an expression")

        model_dict[eq.lhs] = eq.rhs

    return model_dict


class Model:
    def __init__(self, model: dict[sp.Symbol, sp.Expr | Expr] | Iterable[str] | str) -> None:
        if isinstance(model, dict):
            self.model = model
        elif isinstance(model, str):
            self.model = parse_model_str([model])
        elif isinstance(model, Iterable):
            self.model = parse_model_str(model)
        else:
            raise ValueError("Invalid type")

        self.expr: dict = {k: as_expr(v) for k, v in self.model.items()}
        topology = {k: v.symbols for k, v in self.expr.items()}
        self.call_stack = [
            elem for subset in toposort(topology) for elem in subset if elem in self.model.keys()
        ]

    @property
    def x_symbols(self) -> set[sp.Symbol]:
        return set.union(*(v.symbols for v in self.expr.values())) - self.y_symbols

    @property
    def y_symbols(self) -> set[sp.Symbol]:
        return set(self.model.keys())

    def __call__(self, **kwargs):
        resolved = {}
        for key in self.call_stack:
            resolved[key.name] = self.expr[key](**kwargs, **resolved)
        return resolved

    # TODO copy/paste code with Function -> baseclass
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
