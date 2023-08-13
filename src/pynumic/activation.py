"""TODO:"""
import math


class Activation:  # pylint: disable=too-few-public-methods
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

    # ELU, SELU, SWISH, ELiSH

    DEFAULT_ACTIVATION_MODE: int = SIGMOID
    # DEFAULT_ACTIVATION_LIMIT: float = 0.1e-3 # TODO:

    def __init__(self, activation_mode: int) -> None:
        self._activation_mode: int = self.__check_activation_mode(activation_mode)

    @property
    def activation_mode(self) -> int:
        """Activation function mode."""
        return self._activation_mode

    @activation_mode.setter
    def activation_mode(self, value: int) -> None:
        self._activation_mode = self.__check_activation_mode(value)

    def __check_activation_mode(self, value: int) -> int:
        return (
            self.DEFAULT_ACTIVATION_MODE
            if value < self.LINEAR or value > self.TANH
            else value
        )

    def _get_activation(self, value: float) -> float:
        """Activation function."""
        return get_activation(value, self._activation_mode)

    def _get_derivative(self, value: float) -> float:
        """Derivative activation function."""
        return get_derivative(value, self._activation_mode)


def get_activation(value: float, mode: int = Activation.SIGMOID) -> float:
    """Activation function."""
    match mode:
        case Activation.LINEAR:
            # if math.fabs(value) > 1.:  # TODO:
            #     return 0.01 * value
            #     #return math.copysign(0.9, value)  # TODO:
            return value
        case Activation.RELU:
            return 0 if value < 0 else value
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
            return 1 - value ** 2
        case Activation.SIGMOID | _:
            return value * (1 - value)
