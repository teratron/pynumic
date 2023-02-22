"""Properties of neural network."""

import random
from typing import Any

from pynumic.interface.initialize import initialize
from pynumic.properties.activation import Activation
from pynumic.properties.bias import Bias
from pynumic.properties.layers import Layers
from pynumic.properties.loss import Loss
from pynumic.properties.rate import Rate


class Neuron:
    def __init__(self, value: float, miss: float) -> None:
        self.value = value
        self.miss = miss


class Properties(
    Bias,
    Layers,
    Activation,
    Loss,
    Rate
):
    """Properties of neural network."""

    def __init__(self, **props: Any) -> None:
        # Weights
        if "weights" in props:
            self.weights = props["weights"]
            del props["weights"]
        else:
            self.weights = [
                [[random.uniform(-0.5, 0.5) for _ in range(5)]
                 for _ in range(5)]
                for _ in range(5)
            ]

        # Config
        if "config" in props:
            self.config = props["config"]
            del props["config"]

        # Layers.__init__(self, hidden_layers)
        # Activation.__init__(self, activation_mode)
        # Loss.__init__(self, loss_mode, loss_limit)
        # Rate.__init__(self, rate)
        # Bias.__init__(self, bias)

        self.__dict__ = props

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
