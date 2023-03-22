"""TODO: Propagation"""
import math

from pynumic.initialization import Initialization


# from dataclasses import dataclass


# @dataclass
# class Neuron:
#     """Neuron."""
#
#     value: float
#     miss: float


class Propagation(Initialization):
    """Propagation."""

    # neurons: list[list[Neuron]]
    _data_input: list[float]
    # _data_output: list[float]
    _data_target: list[float]

    # def _calc_neurons(self, data_input: list[float]) -> None:
    def _calc_neurons(self) -> None:
        """Calculating neurons."""
        dec, length = 0, self._len_input  # self.layers["len_input"]
        for i, layer in enumerate(self.neurons):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j, _ in enumerate(layer):
                # self.__get_neuron(i, j, dec, length, self._data_input)
                self.__get_neuron(i, j, dec, length)

                # if i == self._last_ind:
                #     self._data_output[j] = self.neurons[i][j].value

    def __get_neuron(
            self, i: int, j: int, dec: int, length: int  # , data_input: list[float]
    ) -> None:
        k = self.neurons[i][j].value = 0
        for k, weight in enumerate(self._weights[i][j]):
            if k < length:
                # self.neurons[i][j].value += (
                #     self.neurons[dec][k].value * weight
                #     if i > 0
                #     else data_input[k] * weight
                # )
                if i > 0:
                    self.neurons[i][j].value += self.neurons[dec][k].value * weight
                else:
                    self.neurons[i][j].value += self._data_input[k] * weight
            else:
                self.neurons[i][j].value += weight

        if self._activation_mode == self.LINEAR:
            if k > 0:
                self.neurons[i][j].value /= k
        else:
            self.neurons[i][j].value = self._get_activation(self.neurons[i][j].value)

    # def _calc_loss(self, data_target: list[float]) -> float:
    def _calc_loss(self) -> float:
        """Calculating and return the total error of the output neurons."""
        loss = 0.0
        for i, neuron in enumerate(self.neurons[self._last_ind]):
            # neuron.miss = data_target[i] - neuron.value
            neuron.miss = self._data_target[i] - neuron.value
            match self._loss_mode:
                case self.AVG:
                    loss += math.fabs(neuron.miss)
                case self.ARCTAN:
                    loss += math.atan(neuron.miss) ** 2
                case self.MSE | self.RMSE | _:
                    loss += neuron.miss ** 2

        if math.isnan(loss):  # TODO:
            raise ValueError(f"1 {__name__}: loss not-a-number value")

        if math.isinf(loss):  # TODO:
            raise ValueError(f"1 {__name__}: loss is infinity")

        loss /= self._len_output
        if self._loss_mode == self.RMSE:
            loss = math.sqrt(loss)

        if math.isnan(loss):
            raise ValueError(f"{__name__}: loss not-a-number value")

        if math.isinf(loss):
            raise ValueError(f"{__name__}: loss is infinity")

        return loss

    def _calc_miss(self) -> None:
        """Calculating the error of neuron in hidden layers."""
        for i in range(self._prev_ind, -1, -1):
            inc = i + 1
            for j, _ in enumerate(self.neurons[i]):
                self.neurons[i][j].miss = 0
                for k, _ in enumerate(self.neurons[inc]):
                    self.neurons[i][j].miss += (
                            self.neurons[inc][k].miss * self._weights[inc][k][j]
                    )

    # def _update_weights(self, data_input: list[float]) -> None:
    def _update_weights(self) -> None:
        """Update weights."""
        dec, length = 0, self._len_input  # self.layers["len_input"]
        for i, weight in enumerate(self._weights):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j, _ in enumerate(weight):
                # self.__get_weight(i, j, dec, length, data_input)
                self.__get_weight(i, j, dec, length)

    def __get_weight(
            self, i: int, j: int, dec: int, length: int  # , data_input: list[float]
    ) -> None:
        grad = (
                self._rate
                * self.neurons[i][j].miss
                * self._get_derivative(self.neurons[i][j].value)
        )
        for k, _ in enumerate(self._weights[i][j]):
            if k < length:
                # value = self.neurons[dec][k].value if i > 0 else data_input[k]
                value = self.neurons[dec][k].value if i > 0 else self._data_input[k]

                if self._activation_mode == self.LINEAR:
                    if value != 0:
                        self._weights[i][j][k] += grad / value
                    # self._weights[i][j][k] += grad / value if value != 0 else 0
                else:
                    self._weights[i][j][k] += grad * value
                # self._weights[i][j][k] += grad * value
            else:
                self._weights[i][j][k] += grad

# if __name__ == "__main__":
#     ner = [1, 2, 3, 4]
#     for i, n in enumerate(ner):
#         print(i, n)

# // calcNeuron.
# func (nn *NN) calcNeuron() {
# 	var length, dec int
# 	for i, v := range nn.neuron {
# 		if i > 0 {
# 			dec = i - 1
# 			length = len(nn.neuron[dec])
# 		} else {
# 			length = nn.lenInput
# 		}
#
# 		for j, n := range v {
# 			var num pkg.FloatType = 0
# 			n.value = 0
# 			for k, w := range nn.weights[i][j] {
# 				if k < length {
# 					if i > 0 {
# 						n.value += nn.neuron[dec][k].value * w
# 					} else {
# 						n.value += nn.input[k] * w
# 					}
# 				} else {
# 					n.value += w
# 				}
# 				num++
# 			}
#
# 			switch nn.ActivationMode {
# 			case params.LINEAR:
# 				if num > 0 {
# 					n.value /= num
# 				}
# 			default:
# 				n.value = params.Activation(n.value, nn.ActivationMode)
# 			}
# 		}
# 	}
# }
#
# // calcMiss calculating the error of neuron in hidden layers.
# func (nn *NN) calcMiss() {
# 	if nn.lastLayerIndex > 0 {
# 		for i := nn.lastLayerIndex - 1; i >= 0; i-- {
# 			inc := i + 1
# 			for j, n := range nn.neuron[i] {
# 				n.miss = 0
# 				for k, m := range nn.neuron[inc] {
# 					n.miss += m.miss * nn.weights[inc][k][j]
# 				}
# 			}
# 		}
# 	}
# }
#
# // updateWeight update weights.
# func (nn *NN) updateWeight() {
# 	var length, dec int
# 	for i, v := range nn.weights {
# 		if i > 0 {
# 			dec = i - 1
# 			length = len(nn.neuron[dec])
# 		} else {
# 			length = nn.lenInput
# 		}
#
# 		for j, w := range v {
# 			grad := nn.Rate * nn.neuron[i][j].miss * \
# 			params.Derivative(nn.neuron[i][j].value, nn.ActivationMode)
# 			for k := range w {
# 				if k < length {
# 					var value pkg.FloatType
# 					if i > 0 {
# 						value = nn.neuron[dec][k].value
# 					} else {
# 						value = nn.input[k]
# 					}
#
# 					switch nn.ActivationMode {
# 					case params.LINEAR:
# 						if value != 0 {
# 							nn.weights[i][j][k] += grad / value
# 						}
# 					default:
# 						nn.weights[i][j][k] += grad * value
# 					}
# 				} else {
# 					nn.weights[i][j][k] += grad
# 				}
# 			}
# 		}
# 	}
# }
