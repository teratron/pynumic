"""Properties of neural network."""

import random
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeAlias

from pynumic.activation import Activation
from pynumic.initialize import initialize
from pynumic.loss import Loss


# class Neuron:
#     """TODO:"""
#
#     def __init__(self, value: float, miss: float) -> None:
#         self.value: float = value
#         self.miss: float = miss

@dataclass
class Neuron:
    """Neuron."""
    value: float
    miss: float


NeuronsType: TypeAlias = list[list[Neuron]]
WeightsType: TypeAlias = list[list[list[float]]]
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

    # Neurons
    neurons: NeuronsType = [[Neuron(0, 0)]]

    # Transfer data
    data_weight: WeightsType
    # data_input: list[float]
    # data_target: list[float]
    # data_output: list[float]

    # Settings
    len_input: int = 0
    len_output: int = 0
    last_layer_ind: int = 0
    is_init: bool = False
    is_query: bool = False

    def __init__(
            self,
            *,
            bias: bool = True,
            hidden_layers: LayersType = None,
            activation_mode: int = Activation.DEFAULT_ACTIVATION_MODE,
            loss_mode: int = Loss.DEFAULT_LOSS_MODE,
            loss_limit: float = Loss.DEFAULT_LOSS_LIMIT,
            rate: float = DEFAULT_RATE,
            weights: WeightsType | None = None
    ) -> None:
        self._bias: bool = bias
        self.hidden_layers: LayersType = self.__check_layers(hidden_layers)
        self._rate: float = self.__check_rate(rate)
        # self.weights = weights

        # Weights
        if weights is not None:
            self.weights: WeightsType = weights
        else:
            self.weights = [
                [
                    [
                        random.uniform(-0.5, 0.5) for _ in range(5)
                    ] for _ in range(5)
                ] for _ in range(5)
            ]

        self.data_input: list[float] = [0 for _ in range(2)]
        self.data_target: list[float] = [0 for _ in range(2)]
        self.data_output: list[float] = [0 for _ in range(2)]
        self.mutex = Lock()

        Activation.__init__(self, activation_mode)
        Loss.__init__(self, loss_mode, loss_limit)

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

        if self._hidden_layers[0] > 0:
            self.last_layer_ind = len(self._hidden_layers)
            # self.layers = self._hidden_layers.append(self.len_output)
        else:
            self.last_layer_ind = 0

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
