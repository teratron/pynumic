"""TODO:"""
import math

from typing import Callable, Iterable, Union, TypeAlias, Any


class Loss:
    """Loss.

    The mode of calculation of the total error:

    * MSE -- Mean Squared Error (0);
    * RMSE -- Root Mean Squared Error (1);
    * ARCTAN -- Arctan Error (2);
    * AVG -- Average Error (3).
    """

    MSE: int = 0
    """MSE -- Mean Squared Error (0)."""

    RMSE: int = 1
    """RMSE -- Root Mean Squared Error (1)."""

    ARCTAN: int = 2
    """ARCTAN -- Arctan Error (2)."""

    AVG: int = 3
    """AVG -- Average Error (3)."""

    DEFAULT_LOSS_MODE: int = MSE
    DEFAULT_LOSS_LIMIT: float = 1e-10

    def __init__(self, loss_mode: int, loss_limit: float) -> None:
        self._loss_mode: int = self.__check_loss_mode(loss_mode)
        self._loss_limit: float = self.__check_loss_limit(loss_limit)

    @property
    def loss_mode(self) -> int:
        """The mode of calculation of the total error."""
        return self._loss_mode

    @loss_mode.setter
    def loss_mode(self, value: int) -> None:
        self._loss_mode = self.__check_loss_mode(value)

    def __check_loss_mode(self, value: int) -> int:
        return (
            self.DEFAULT_LOSS_MODE
            if value < self.MSE or value > self.AVG
            else value
        )

    @property
    def loss_limit(self) -> float:
        """Minimum (sufficient) limit of the average of the error during training."""
        return self._loss_limit

    @loss_limit.setter
    def loss_limit(self, value: float) -> None:
        self._loss_limit = self.__check_loss_limit(value)

    def __check_loss_limit(self, value: float) -> float:
        return self.DEFAULT_LOSS_LIMIT if value < 0 else value


_TargetType: TypeAlias = Callable[[Any], Union[Iterable[float], float]]
_InnerType: TypeAlias = Callable[[Any], float]


def _total_loss(func: _TargetType) -> _InnerType:
    def inner(obj: Any) -> float:
        loss = 0.0
        miss = func(obj)
        if isinstance(miss, Iterable):
            count = 0.0
            for value in miss:
                loss += __get_loss(value, obj.loss_mode)
                count += 1

            if count > 1:
                loss /= count
        elif isinstance(miss, float):
            loss += __get_loss(miss, obj.loss_mode)

        if math.isnan(loss):
            raise ValueError(f'{__name__}: loss not-a-number value')

        if math.isinf(loss):
            raise ValueError(f'{__name__}: loss is infinity')

        if obj.loss_mode == Loss.RMSE:
            loss = math.sqrt(loss)

        return loss
    return inner


def __get_loss(value: float, mode: int) -> float:
    match mode:
        case Loss.AVG:
            return math.fabs(value)
        case Loss.ARCTAN:
            return math.atan(value) ** 2
        case Loss.MSE | Loss.RMSE | _:
            return value ** 2

# @_total_loss(3)
# def calc_loss() -> Iterable[float]:
#     for value in (0.1, 0.2, 0.3, 0.4):
#         yield value
#
#
# @_total_loss(2)
# def _calc_loss() -> float:
#     return 0.333
#
#
# if __name__ == "__main__":
#     print('calc_loss', type(calc_loss()))
#     print('_calc_loss', type(_calc_loss()))
