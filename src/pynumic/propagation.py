"""TODO: Propagation"""
import math

from src.pynumic.initialization import Initialization


class Propagation(Initialization):
    """Propagation."""

    def calc_neurons(self, data_input: list[float]) -> None:
        """Calculating neurons."""
        _dec, _len = 0, self.layers["len_input"]
        for i, layer in enumerate(self.neurons):
            if i > 0:
                _dec = i - 1
                _len = len(self.neurons[_dec])

            for j, _ in enumerate(layer):
                self.__get_neuron(i, j, _dec, _len, data_input)

    def __get_neuron(
            self, i: int, j: int, _dec: int, _len: int, data_input: list[float]
    ) -> None:
        k = self.neurons[i][j].value = 0
        for k, weight in enumerate(self.weights[i][j]):
            if k < _len:
                self.neurons[i][j].value += (
                    self.neurons[_dec][k].value * weight
                    if i > 0
                    else data_input[k] * weight
                )
            else:
                self.neurons[i][j].value += weight

        if self.activation_mode == self.LINEAR:
            self.neurons[i][j].value /= k if k > 0 else 1
        else:
            self.neurons[i][j].value = self.get_activation(
                    self.neurons[i][j].value
            )

    def calc_loss(self, data_target: list[float]) -> float:
        """Calculating and return the total error of the output neurons."""
        loss = 0.0
        for i, neuron in enumerate(self.neurons[self.layers["last_index"]]):
            neuron.miss = data_target[i] - neuron.value
            match self.loss_mode:
                case self.AVG:
                    loss += math.fabs(neuron.miss)
                case self.ARCTAN:
                    loss += math.atan(neuron.miss) ** 2
                case self.MSE | self.RMSE | _:
                    loss += neuron.miss ** 2

        loss /= self.layers["len_output"]
        if self.loss_mode == self.RMSE:
            loss = math.sqrt(loss)

        if math.isnan(loss):
            raise ValueError(f"{__name__}: loss not-a-number value")

        if math.isinf(loss):
            raise ValueError(f"{__name__}: loss is infinity")

        return loss

    def calc_miss(self) -> None:
        """Calculating the error of neuron in hidden layers."""
        for i in range(self.layers["prev_index"], -1, -1):
            inc = i + 1
            for j, _ in enumerate(self.neurons[i]):
                self.neurons[i][j].miss = 0
                for k, _ in enumerate(self.neurons[inc]):
                    self.neurons[i][j].miss += (
                            self.neurons[inc][k].miss * self.weights[inc][k][j]
                    )

    def update_weights(self, data_input: list[float]) -> None:
        """Update weights."""
        _dec, _len = 0, self.layers["len_input"]
        if self.weights is not None:
            for i, weight in enumerate(self.weights):
                if i > 0:
                    _dec = i - 1
                    _len = len(self.neurons[_dec])

                for j, _ in enumerate(weight):
                    self.__get_weight(i, j, _dec, _len, data_input)

    def __get_weight(
            self, i: int, j: int, _dec: int, _len: int, data_input: list[float]
    ) -> None:
        grad = (
                self.rate
                * self.neurons[i][j].miss
                * self.get_derivative(self.neurons[i][j].value)
        )
        for k, _ in enumerate(self.weights[i][j]):
            if k < _len:
                value = self.neurons[_dec][k].value if i > 0 else data_input[k]

                if self.activation_mode == self.LINEAR:
                    self.weights[i][j][k] += grad / value if value != 0 else 0
                else:
                    self.weights[i][j][k] += grad * value
            else:
                self.weights[i][j][k] += grad

# if __name__ == "__main__":
#     for i in range(-1, -1, -1):
#         print(i)

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
# // calcLoss calculating the error of the output neuron.
# func (nn *NN) calcLoss() (loss float64) {
# 	for i, n := range nn.neuron[nn.lastLayerIndex] {
# 		n.miss = nn.target[i] - n.value
# 		switch nn.LossMode {
# 		default:
# 			fallthrough
# 		case params.MSE, params.RMSE:
# 			loss += math.Pow(float64(n.miss), 2)
# 		case params.ARCTAN:
# 			loss += math.Pow(math.Atan(float64(n.miss)), 2)
# 		case params.AVG:
# 			loss += math.Abs(float64(n.miss))
# 		}
# 	}
#
# 	loss /= float64(nn.lenOutput)
# 	if nn.LossMode == params.RMSE {
# 		loss = math.Sqrt(loss)
# 	}
#
# 	switch {
# 	case math.IsNaN(loss):
# 		log.Panic("perceptron.NN.calcLoss: loss not-a-number value") // TODO: log.Panic (?)
# 	case math.IsInf(loss, 0):
# 		log.Panic("perceptron.NN.calcLoss: loss is infinity") // TODO: log.Panic (?)
# 	}
# 	return
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
# 			grad := nn.Rate * nn.neuron[i][j].miss * params.Derivative(nn.neuron[i][j].value, nn.ActivationMode)
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
