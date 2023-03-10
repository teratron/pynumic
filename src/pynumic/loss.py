"""TODO:"""


class Loss:
    """Loss.

    The mode of calculation of the total error:

    * MSE -- Mean Squared Error (0);
    * RMSE -- Root Mean Squared Error (1);
    * ARCTAN -- Arctan Error (2);
    * AVG -- Average Error (3).
    """

    # __slots__ = ("_loss_mode", "_loss_limit")

    MSE: int = 0
    """MSE -- Mean Squared Error (0)."""

    RMSE: int = 1
    """RMSE -- Root Mean Squared Error (1)."""

    ARCTAN: int = 2
    """ARCTAN -- Arctan Error (2)."""

    AVG: int = 3
    """AVG -- Average Error (3)."""

    DEFAULT_LOSS_MODE: int = MSE
    DEFAULT_LOSS_LIMIT: float = 0.1e-3

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
        return self.DEFAULT_LOSS_LIMIT if value <= 0 else value
