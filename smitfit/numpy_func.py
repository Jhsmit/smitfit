import numpy as np

from smitfit.composite_expr import FuncExpr


def __getattr__(name: str):
    ufunc = getattr(np, name, None)
    if callable(ufunc):

        def lazy_ufunc(*args, **kwargs):
            return FuncExpr(ufunc, *args, **kwargs)

        return lazy_ufunc
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
