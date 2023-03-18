import pytest

# class TestActivation:
#     def test_activation(self):
#         assert False
from pynumic.activation import Activation, get_activation, get_derivative


# from pynumic.activation import Activation, get_activation, get_derivative


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
    assert get_activation(value, mode) == result


@pytest.mark.parametrize(
        "value, mode, result", [
            (0.1, Activation.LINEAR, 1),
            (0.1, Activation.RELU, 1),
            (-0.1, Activation.RELU, 0),
            (0.1, Activation.LEAKY_RELU, 1),
            (-0.1, Activation.LEAKY_RELU, 0.01),
            (0.1, Activation.SIGMOID, 0.09000000000000001),
            (0.1, Activation.TANH, 0.99),
            (0.1, 255, 0.09000000000000001)
        ]
        )
def test_get_derivative(value: float, mode: int, result: float) -> None:
    assert get_derivative(value, mode) == result

# 		{
# 			name: "#1_LINEAR",
# 			args: args{.1, LINEAR},
# 			want: 1,
# 		},
# 		{
# 			name: "#2_RELU",
# 			args: args{.1, RELU},
# 			want: 1,
# 		},
# 		{
# 			name: "#3_RELU",
# 			args: args{-.1, RELU},
# 			want: 0,
# 		},
# 		{
# 			name: "#4_LEAKYRELU",
# 			args: args{.1, LEAKYRELU},
# 			want: 1,
# 		},
# 		{
# 			name: "#5_LEAKYRELU",
# 			args: args{-.1, LEAKYRELU},
# 			want: .01,
# 		},
# 		{
# 			name: "#6_SIGMOID",
# 			args: args{.1, SIGMOID},
# 			want: .089999996,
# 		},
# 		{
# 			name: "#7_TANH",
# 			args: args{.1, TANH},
# 			want: .99,
# 		},
# 		{
# 			name: "#8_default",
# 			args: args{.1, 255},
# 			want: .089999996,
# 		},
