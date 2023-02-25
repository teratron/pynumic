"""TODO:"""
import math

from src.pynumic.properties import Properties


class Propagation(Properties):
    """Propagation."""

    _len: int
    _dec: int

    def _calc_neurons(self) -> None:
        """Calculating neurons."""
        i, self._dec, self._len = 0, 0, self.len_input
        for neuron in self.neurons:
            if i > 0:
                self._dec = i - 1
                self._len = len(self.neurons[self._dec])

            for j in range(len(neuron)):
                self._get_neuron(i, j)
            i += 1

    def _get_neuron(self, i: int, j: int) -> None:
        k = self.neurons[i][j].value = 0
        for weight in self.data_weight[i][j]:
            if k < self._len:
                if i > 0:
                    self.neurons[i][j].value += self.neurons[self._dec][k].value * weight
                else:
                    self.neurons[i][j].value += self.data_input[k] * weight
            else:
                self.neurons[i][j].value += weight
            k += 1

        if self.activation_mode == self.LINEAR:
            if k > 0:
                self.neurons[i][j].value /= k
        else:
            self.neurons[i][j].value = self.get_activation(self.neurons[i][j].value)

    def _calc_loss(self) -> float:
        """Calculating and return the total error of the output neurons."""
        i, error = 0, 0.0
        for neuron in self.neurons[self.last_layer_ind]:
            neuron.miss = self.data_target[i] - neuron.value
            match self.loss_mode:
                case self.AVG:
                    error += math.fabs(neuron.miss)
                case self.ARCTAN:
                    error += math.atan(neuron.miss) ** 2
                case self.MSE | self.RMSE | _:
                    error += neuron.miss ** 2
            i += 1

        error /= self.len_output
        if self.loss_mode == self.RMSE:
            error = math.sqrt(error)

        if math.isnan(error):
            raise ValueError(f"{__name__}: loss not-a-number value")

        if math.isinf(error):
            raise ValueError(f"{__name__}: loss is infinity")

        return error

    def _calc_miss(self) -> None:
        """Calculating the error of neuron in hidden layers."""
        if self.last_layer_ind > 0:
            for i in range(self.last_layer_ind - 1, -1, -1):
                inc = i + 1
                for j in range(len(self.neurons[i])):
                    self.neurons[i][j].miss = 0
                    for k in range(len(self.neurons[inc])):
                        self.neurons[i][j].miss += self.neurons[inc][k].miss * self.data_weight[inc][k][j]

    def _update_weights(self) -> None:
        """Update weights."""
        i, self._dec, self._len = 0, 0, self.len_input
        for weight in self.data_weight:
            if i > 0:
                self._dec = i - 1
                self._len = len(self.neurons[self._dec])

            for j in range(len(weight)):
                self._get_weight(i, j)
            i += 1

    def _get_weight(self, i: int, j: int) -> None:
        grad = self.rate * self.neurons[i][j].miss * self.get_derivative(self.neurons[i][j].value)
        for k in range(len(self.data_weight[i][j])):
            if k < self._len:
                value = self.neurons[self._dec][k].value
                if i == 0:
                    value = self.data_input[k]

                if self.activation_mode == self.LINEAR:
                    if value != 0:
                        self.data_weight[i][j][k] += grad / value
                else:
                    self.data_weight[i][j][k] += grad * value
            else:
                self.data_weight[i][j][k] += grad


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
