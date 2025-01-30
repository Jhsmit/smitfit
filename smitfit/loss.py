from typing import Optional

from smitfit.model import Model
from smitfit.reduce import mean_reduction, sum_reduction


class Loss:
    """sum/average reduction"""

    def __call__(self, **kwargs) -> float:
        return 0.0


class SELoss(Loss):
    """sum of squared errors"""

    def __init__(self, model: Model, y_data: dict, weights: Optional[dict] = None):
        self.model = model
        self.y_data = y_data
        self.weights = weights or {}

    def squares(self, **kwargs):
        y_model = self.model(**kwargs)

        squares = {
            k: ((y_model[k] - self.y_data[k]) * self.weights.get(k, 1)) ** 2
            for k in self.y_data.keys()
        }

        return squares

    def __call__(self, **kwargs) -> float:
        squares = self.squares(**kwargs)

        return sum_reduction(squares)


class MSELoss(SELoss):
    def __call__(self, **kwargs) -> float:
        squares = self.squares(**kwargs)

        return mean_reduction(squares)
