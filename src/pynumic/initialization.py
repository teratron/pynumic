"""TODO: Initialization."""
import random
from dataclasses import dataclass

from pynumic.properties import Properties


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


@dataclass
class Length:
    """Length."""

    input: int
    output: int


@dataclass
class Index:
    """Index."""

    last: int
    prev: int


class Initialization(Properties):
    """initialization neural network."""

    __slots__ = (
        "_neurons",
        "_len",
        "_ind"
    )

    _neurons: list[list[Neuron]]
    _len: Length
    _ind: Index

    def _init(self, len_input: int = 0, len_target: int = 0) -> bool:
        is_init: bool = False
        self._len = Length(0, 0)
        self._ind = Index(0, 0)

        if self._weights:
            is_init = self.__init_from_weight()
        elif len_input > 0 and len_target > 0:
            is_init = self.__init_from_new(len_input, len_target)

        return is_init

    def __init_from_new(self, len_input: int, len_target: int) -> bool:
        self._len.input = len_input
        self._len.output = len_target

        weights: list[int] = [self._len.input + int(self._bias)]
        layers: list[int] = [self._len.output]
        if self._hidden_layers:
            self._ind.last = len(self._hidden_layers)
            weights += list(map(lambda x: x + int(self._bias), self._hidden_layers))
            layers = self._hidden_layers + layers

        self._ind.prev = self._ind.last - 1
        self._weights = [
            [
                [
                    -0.5 if self._activation_mode == self.LINEAR
                    else round(random.uniform(-0.5, 0.5), 3)
                    for _ in range(weights[i])
                ] for _ in range(v)
            ] for i, v in enumerate(layers)
        ]
        self._neurons = [[Neuron(0, 0) for _ in range(v)] for v in layers]
        del weights, layers
        return True

    def __init_from_weight(self) -> bool:
        length = len(self._weights)
        self._ind.last = length - 1
        self._ind.prev = self._ind.last - 1
        self._len.input = len(self._weights[0][0])
        self._len.output = len(self._weights[self._ind.last])

        if length > 1 and len(self._weights[0]) + 1 == len(self._weights[1][0]):
            self._bias = True
            self._len.input -= 1

        if self._ind.last > 0:
            self._hidden_layers = [
                len(self._weights[i]) for i, _ in enumerate(self._hidden_layers)
            ]

        self._neurons = [[Neuron(0, 0) for _ in v] for v in self._weights]
        return True
