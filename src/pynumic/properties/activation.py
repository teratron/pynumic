"""TODO:"""

import math


class Activation:
    """Activation.

    Mode:

    * LINEAR -- Linear/identity (0);
    * RELU -- ReLu (rectified linear unit) (1);
    * LEAKY_RELU -- Leaky ReLu (leaky rectified linear unit) (2);
    * SIGMOID -- Logistic, a.k.a. sigmoid or soft step (3);
    * TANH -- TanH (hyperbolic tangent) (4).
    """

    LINEAR: int = 0
    """LINEAR -- Linear/identity (0)."""

    RELU: int = 1
    """RELU -- ReLu (rectified linear unit) (1)."""

    LEAKY_RELU: int = 2
    """LEAKY_RELU -- Leaky ReLu (leaky rectified linear unit) (2)."""

    SIGMOID: int = 3
    """SIGMOID -- Logistic, a.k.a. sigmoid or soft step (3)."""

    TANH: int = 4
    """TANH -- TanH (hyperbolic tangent) (4)."""

    _activation_mode: int = SIGMOID

    @property
    def activation_mode(self) -> int:
        """TODO:"""
        return self._activation_mode

    @activation_mode.setter
    def activation_mode(self, value: int) -> None:
        self._activation_mode = Activation.check_activation_mode(value)

    @classmethod
    def check_activation_mode(cls, value: int) -> int:
        """Checking whether the value corresponds to normal conditions."""
        return cls.SIGMOID if value < cls.LINEAR or value > cls.TANH else value

    def get_activation(self, value: float) -> float:
        """Activation function."""
        return get_activation(value, self._activation_mode)

    def get_derivative(self, value: float) -> float:
        """Derivative activation function."""
        return get_derivative(value, self._activation_mode)


def get_activation(value: float, mode: int = Activation.SIGMOID) -> float:
    """Activation function."""
    match mode:
        case Activation.LINEAR:
            return value
        case Activation.RELU:
            return 0.0 if value < 0 else value
        case Activation.LEAKY_RELU:
            return 0.01 * value if value < 0 else value
        case Activation.TANH:
            value = math.exp(2 * value)
            return (value - 1) / (value + 1)
        case Activation.SIGMOID | _:
            return 1 / (1 + math.exp(-value))


def get_derivative(value: float, mode: int = Activation.SIGMOID) -> float:
    """Derivative activation function."""
    match mode:
        case Activation.LINEAR:
            return 1
        case Activation.RELU:
            return 0 if value < 0 else 1
        case Activation.LEAKY_RELU:
            return 0.01 if value < 0 else 1
        case Activation.TANH:
            return 1 - value**2
        case Activation.SIGMOID | _:
            return value * (1 - value)
