"""TODO: properties.py -"""
from dataclasses import dataclass
from typing import TypeAlias

from pynumic.activation import Activation
from pynumic.loss import Loss
from pynumic.parameters import Parameters

WeightsType: TypeAlias = list[list[list[float]]]


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


class Properties(Activation, Loss):
    """Properties of neural network."""

    __slots__ = (
        "_params",
        "_neurons"
    )

    DEFAULT_RATE: float = 0.3
    _neurons: list[list[Neuron]]

    def __init__(
            self,
            *,
            bias: bool = False,
            hidden_layers: list[int] | None = None,
            activation_mode: int = Activation.DEFAULT_ACTIVATION_MODE,
            loss_mode: int = Loss.DEFAULT_LOSS_MODE,
            loss_limit: float = Loss.DEFAULT_LOSS_LIMIT,
            rate: float = DEFAULT_RATE,
            weights: WeightsType | None = None,
    ) -> None:
        self._params = Parameters()
        self._bias: bool = bias
        self._hidden_layers: list[int] = self.__check_hidden_layers(hidden_layers)
        self._rate: float = self.__check_rate(rate)
        self._weights: WeightsType = self.__check_weights(weights)

        Activation.__init__(self, activation_mode)
        Loss.__init__(self, loss_mode, loss_limit)

    ############################################################################
    # Bias
    ############################################################################
    @property
    def bias(self) -> bool:
        """The neuron bias, false or true (required field for a config)."""
        return self._bias

    @bias.setter
    def bias(self, value: bool) -> None:
        self._bias = value

    ############################################################################
    # Hidden Layers
    ############################################################################
    @property
    def hidden_layers(self) -> list[int]:
        """List of the number of neuron in each hidden layers."""
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value: list[int]) -> None:
        self._hidden_layers = self.__check_hidden_layers(value)

    @staticmethod
    def __check_hidden_layers(value: list[int] | None) -> list[int]:
        if not value or value is None or value == [0]:
            return []

        if isinstance(value, list) and all(list(map(lambda i: i > 0, value))):
            return value

        raise ValueError(f"{__name__}: array of hidden layers incorrectly set {value}")

    ############################################################################
    # Rate
    ############################################################################
    @property
    def rate(self) -> float:
        """Learning coefficient (greater than 0.0 and less than or equal to 1.0)."""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate = self.__check_rate(value)

    def __check_rate(self, value: float) -> float:
        return self.DEFAULT_RATE if value <= 0 or value > 1 else value

    ############################################################################
    # Weights
    ############################################################################
    @property
    def weights(self) -> WeightsType:
        """TODO:"""
        return self._weights

    @weights.setter
    def weights(self, value: WeightsType) -> None:
        self._weights = self.__check_weights(value)

    def __check_weights(self, value: WeightsType | None) -> WeightsType:
        if not value or value is None or value == [[[0]]]:
            return []

        if isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], list):
            if self.__init_weights(value):
                return value

        raise ValueError(f"{__name__}: array of weights incorrectly set {value}")

    def __init_weights(self, value: WeightsType) -> bool:
        length = len(value)
        self._params.last_ind = length - 1
        self._params.len_input = len(value[0][0])
        self._params.len_output = len(value[self._params.last_ind])

        if length > 1 and len(value[0]) + 1 == len(value[1][0]):
            self._bias = True
            self._params.len_input -= 1

        # print(self.bias, length, self._params.last_ind, self._params.prev_ind)
        print(self._params.len_input, self._params.len_output)

        if self._params.last_ind > 0:
            self._hidden_layers = [
                len(value[i]) for i, _ in enumerate(value)
            ]
            # TODO:
            print(self._hidden_layers)
            self._params.layers = (
                    [self._params.len_output]
                    + self._hidden_layers
                    + [self._params.len_output]
            )
            print(self._params.layers)

        self._neurons = [[Neuron(0, 0) for _ in v] for v in value]
        self._params.is_init = True

        return self._params.is_init
