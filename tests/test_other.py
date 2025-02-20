from smitfit.model import Model
import sympy as sp


def test_model_from_str():
    model = Model("y == a * x + b")
    assert model.y_symbols == {sp.Symbol("y")}
    assert model.x_symbols == {sp.Symbol("x"), sp.Symbol("a"), sp.Symbol("b")}
    assert model.model[sp.Symbol("y")] == sp.Add(
        sp.Mul(sp.Symbol("a"), sp.Symbol("x")), sp.Symbol("b")
    )
