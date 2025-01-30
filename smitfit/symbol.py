import numpy as np
import numpy.typing as npt
import sympy as sp
from smitfit.utils import Optional


class Symbols(dict):
    def __init__(self, names, cls=sp.Symbol, **kwargs):
        default_kwargs = {"seq": True}
        default_kwargs.update(kwargs)
        super().__init__({s.name: s for s in sp.symbols(names, cls=cls, **default_kwargs)})

    def __repr__(self):
        return f"Symbols({list(self.keys())})"

    def __getattr__(self, name) -> sp.Symbol:
        if name in self:
            return self[name]
        raise AttributeError(f"'SymbolNamespace' object has no attribute '{name}'")


def symbol_matrix(
    name: Optional[str] = None,
    shape: Optional[tuple[int, ...]] = None,
    names: Optional[npt.ArrayLike] = None,
    suffix: Optional[npt.ArrayLike] = None,
) -> sp.Matrix:
    if shape is None:
        if names is not None:
            shape = (len(names), 1)
        elif suffix is not None:
            shape = (len(suffix), 1)
        else:
            raise ValueError("If 'shape' is not given, must specify 'names' or 'suffix'")

    # Generate names for parameters. Uses 'names' first, then <name>_<suffix> otherwise generates suffices
    # from indices
    if names is None and name is None:
        raise ValueError("Must specify either 'name' or 'names'")
    elif names is None:
        names = np.full(shape, fill_value="", dtype=object)
        if suffix is None:
            for i, j in np.ndindex(shape):
                names[i, j] = f"{name}_{i}_{j}"
        else:
            suffix = np.array(suffix).reshape(shape)
            for i, j in np.ndindex(shape):
                names[i, j] = f"{name}_{suffix[i, j]}"
    else:
        names = np.array(names)

    matrix = sp.zeros(*shape)
    for i, j in np.ndindex(shape):
        matrix[i, j] = sp.Symbol(
            name=names[i, j],
        )

    return matrix
