import sympy as sp
import numpy as np
from smitfit.parameter import Parameter, Parameters, unpack, pack


def test_parameter_initialization():
    symbol = sp.Symbol("a")
    param = Parameter(symbol, guess=2.0, lower_bound=0, upper_bound=10, fixed=True)

    assert param.symbol == symbol
    assert param.guess == 2.0
    assert param.bounds == (0, 10)
    assert param.fixed is True


def test_parameter_fix_unfix():
    symbol = sp.Symbol("a")
    param = Parameter(symbol)

    param.fix()
    assert param.fixed is True

    param.unfix()
    assert param.fixed is False


def test_parameter_set_bounds():
    symbol = sp.Symbol("a")
    param = Parameter(symbol)

    param.set_bounds(0, 10)
    assert param.bounds == (0, 10)


def test_parameter_set_guess():
    symbol = sp.Symbol("a")
    param = Parameter(symbol)

    param.set_guess(5.0)
    assert param.guess == 5.0


def test_parameters_initialization():
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    params = Parameters.from_symbols(symbols)

    assert len(params) == 2
    assert params["a"].name == "a"
    assert params["b"].name == "b"


def test_parameters_fix_unfix():
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    params = Parameters.from_symbols(symbols)

    params.fix("a")
    assert params["a"].fixed is True
    assert params["b"].fixed is False

    params.unfix("a")
    assert params["a"].fixed is False


def test_parameters_set_bounds():
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    params = Parameters.from_symbols(symbols)

    bounds_dict = {"a": (0, 10), "b": (1, 5)}
    params.set_bounds(bounds_dict)  # type: ignore

    assert params["a"].bounds == (0, 10)
    assert params["b"].bounds == (1, 5)


def test_parameters_set_guesses():
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    params = Parameters.from_symbols(symbols)

    guess_dict = {"a": 2.0, "b": 3.0}
    params.set_guesses(guess_dict)  # type: ignore

    assert params["a"].guess == 2.0
    assert params["b"].guess == 3.0


def test_unpack():
    shapes = {"a": (2,), "b": (4,)}
    x = np.array([1, 2, 3, 4, 5, 6])

    unpacked = unpack(x, shapes)

    assert np.array_equal(unpacked["a"], np.array([1, 2]))
    assert np.array_equal(unpacked["b"], np.array([3, 4, 5, 6]))


def test_pack():
    parameter_values = [np.array([1, 2]), np.array([3, 4, 5, 6])]

    packed = pack(parameter_values)

    assert np.array_equal(packed, np.array([1, 2, 3, 4, 5, 6]))
