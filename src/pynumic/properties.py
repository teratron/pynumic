"""Properties of neural network."""

from typing import Any

from pynumic.activation import Activation
from pynumic.interface.initialize import initialize
from pynumic.loss import Loss

LayersType = list[int] | None


class Neuron:
    """TODO:"""

    def __init__(self, value: float, miss: float) -> None:
        self.value = value
        self.miss = miss


class Properties(Activation, Loss):
    """Properties of neural network."""

    # __slots__ = (
    #     "_bias",
    #     "_hidden_layers",
    #     "_activation_mode",
    #     "_loss_mode",
    #     "_loss_limit",
    #     "_rate"
    # )

    DEFAULT_RATE: float = 0.3

    def __init__(
            self,
            *,
            bias: bool = True,
            hidden_layers: LayersType = None,
            activation_mode: int = Activation.TANH,
            loss_mode: int = Loss.RMSE,
            loss_limit: float = Loss.DEFAULT_LOSS_LIMIT,
            rate: float = DEFAULT_RATE
    ) -> None:
        self._bias: bool = bias
        self._hidden_layers: LayersType = self.__check_layers(hidden_layers)
        self._rate: float = self.__check_rate(rate)

        Activation.__init__(self, activation_mode)
        Loss.__init__(self, loss_mode, loss_limit)

    # Neurons
    neurons: list[list[Neuron]]

    # Transfer data
    data_weight: list[list[list[float]]]
    data_input: list[float]
    data_target: list[float]
    data_output: list[float]

    # Settings
    len_input: int = 0
    len_output: int = 0
    last_layer_ind: int = 0

    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize neural network."""
        initialize(self, *args, **kwargs)

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

    @hidden_layers.deleter
    def hidden_layers(self) -> None:
        del self._hidden_layers

    @staticmethod
    def __check_layers(value: LayersType) -> list[int]:
        return [0] if value is None else value

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
