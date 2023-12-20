"""TODO:"""
import pytest

from pynumic.activation import Activation, get_activation, get_derivative


# class TestActivation:
#   def test_activation(self):
#       assert False


# def test___check_activation_mode(mode: int, result: float) -> None:
#     # active = Activation(mode)
#     assert Activation._Activation__check_activation_mode(mode) == result


@pytest.mark.parametrize(
        "value, mode, result", [
            (0.1, Activation.LINEAR, 0.1),
            (0.1, Activation.RELU, 0.1),
            (-0.1, Activation.RELU, 0),
            (0.1, Activation.LEAKY_RELU, 0.1),
            (-0.1, Activation.LEAKY_RELU, -0.001),
            (0.1, Activation.SIGMOID, 0.52497918747894),
            (0.1, Activation.TANH, 0.09966799462495583),
            (0.1, 255, 0.52497918747894)
        ]
)
def test_get_activation(value: float, mode: int, result: float) -> None:
    """TODO:"""
    assert get_activation(value, mode) == result


@pytest.mark.parametrize(
        "value, mode, result", [
            (0.1, Activation.LINEAR, 1),
            (0.1, Activation.RELU, 1),
            (-0.1, Activation.RELU, 0),
            (0.1, Activation.LEAKY_RELU, 1),
            (-0.1, Activation.LEAKY_RELU, 0.01),
            (0.1, Activation.SIGMOID, 0.09),
            (0.1, Activation.TANH, 0.99),
            (0.1, 255, 0.09)
        ]
)
def test_get_derivative(value: float, mode: int, result: float) -> None:
    """TODO:"""
    assert round(get_derivative(value, mode), 2) == result
