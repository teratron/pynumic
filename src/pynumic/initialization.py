"""TODO: Initialization."""
import random
from dataclasses import dataclass

from pynumic.properties import Properties


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


class Initialization(Properties):
    """initialization neural network."""

    neurons: list[list[Neuron]]
    _len_input: int = 0
    _len_output: int = 0
    _last_ind: int = 0
    _prev_ind: int = 0

    # _layer: dict[str, int] = {
    #     last_ind: 0,
    #     prev_ind: 0
    # }

    def _init(self, len_input: int = 0, len_target: int = 0) -> bool:
        is_init: bool = False
        if self._weights:
            is_init = self.__init_from_weight()
        elif len_input > 0 and len_target > 0:
            is_init = self.__init_from_new(len_input, len_target)

        return is_init

    def __init_from_new(self, len_input: int, len_target: int) -> bool:
        self._len_input = len_input
        self._len_output = len_target

        weights: list[int] = [self._len_input + int(self._bias)]
        layers: list[int] = [self._len_output]
        if self._hidden_layers:
            self._last_ind = len(self._hidden_layers)
            weights += list(map(lambda x: x + int(self._bias), self._hidden_layers))
            layers = self._hidden_layers + layers

        self._prev_ind = self._last_ind - 1
        self._weights = [
            [
                [
                    -0.5 if self._activation_mode == self.LINEAR
                    else round(random.uniform(-0.5, 0.5), 3)
                    for _ in range(weights[i])
                ] for _ in range(v)
            ] for i, v in enumerate(layers)
        ]
        self.neurons = [[Neuron(0, 0) for _ in range(v)] for v in layers]
        del weights, layers
        return True

    def __init_from_weight(self) -> bool:
        length = len(self._weights)
        self._last_ind = length - 1
        self._prev_ind = self._last_ind - 1
        self._len_input = len(self._weights[0][0])
        self._len_output = len(self._weights[self._last_ind])

        if length > 1 and len(self._weights[0]) + 1 == len(self._weights[1][0]):
            self._bias = True
            self._len_input -= 1

        if self._last_ind > 0:
            self._hidden_layers = [
                len(self._weights[i]) for i, _ in enumerate(self._hidden_layers)
            ]

        self.neurons = [[Neuron(0, 0) for _ in v] for v in self._weights]
        return True
