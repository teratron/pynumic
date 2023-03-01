"""TODO: Properties of neural network."""
from typing import TypeAlias

from pynumic.activation import Activation
from pynumic.loss import Loss

WeightsType: TypeAlias = list[list[list[float]]] | None
LayersType: TypeAlias = list[int] | None


class Properties(Activation, Loss):
    """Properties of neural network."""

    # __slots__ = (
    #     "_bias",
    #     "_hidden_layers",
    #     "_activation_mode",
    #     "_loss_mode",
    #     "_loss_limit",
    #     "_rate",
    #     "weights",
    # )

    DEFAULT_RATE: float = 0.3

    def __init__(
            self,
            *,
            bias: bool = False,
            hidden_layers: LayersType = None,  # None, [], [0]
            activation_mode: int = Activation.DEFAULT_ACTIVATION_MODE,
            loss_mode: int = Loss.DEFAULT_LOSS_MODE,
            loss_limit: float = Loss.DEFAULT_LOSS_LIMIT,
            rate: float = DEFAULT_RATE,
            weights: WeightsType = None
    ) -> None:
        self._bias: bool = bias
        self._hidden_layers: LayersType = self.__check_layers(hidden_layers)
        self._rate: float = self.__check_rate(rate)
        self.weights: WeightsType = weights

        Activation.__init__(self, activation_mode)
        Loss.__init__(self, loss_mode, loss_limit)

    # Bias
    @property
    def bias(self) -> bool:
        """The neuron bias, false or true (required field for a config)."""
        return self._bias

    @bias.setter
    def bias(self, value: bool) -> None:
        self._bias = value

    # Hidden Layers
    @property
    def hidden_layers(self) -> LayersType:
        """List of the number of neuron in each hidden layers."""
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value: list[int]) -> None:
        self._hidden_layers = self.__check_layers(value)

    @staticmethod
    def __check_layers(value: LayersType) -> list[int]:
        # TODO:
        return value

    # Rate
    @property
    def rate(self) -> float:
        """Learning coefficient (greater than 0.0 and less than or equal to 1.0)."""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate = self.__check_rate(value)

    def __check_rate(self, value: float) -> float:
        return self.DEFAULT_RATE if value <= 0 or value > 1 else value
