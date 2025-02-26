from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Iterable, cast, Dict, Union, Mapping

import sympy as sp
from toposort import toposort

from smitfit.expr import Expr, as_expr, _parse_subs_args
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
            self.model = cast(dict[sp.Symbol, sp.Expr | Expr], model)
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

    def subs(self, *args, **kwargs) -> Model:
        """
        Substitute symbols in the model with other symbols or expressions.

        Works similar to sympy's subs() method. Returns a new Model instance.

        Args:
            *args: Can be a dict, list, or tuple of (old, new) pairs
            **kwargs: Can be symbol names and their replacements

        Returns:
            A new Model with substituted expressions
        """
        # Get all relevant symbols from the model
        all_symbols = self.x_symbols.union(self.y_symbols)

        # Parse substitution arguments
        subs_dict = _parse_subs_args(*args, symbols=all_symbols, **kwargs)
        # Create new model with substitutions
        new_model = {}
        for symbol, expr in self.model.items():
            if isinstance(expr, (sp.Expr, sp.MatrixBase, Expr)):
                new_expr = expr.subs(subs_dict)
            else:
                new_expr = expr

            new_model[symbol] = new_expr

        return Model(new_model)
