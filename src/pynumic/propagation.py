from typing import Iterable

from pynumic.properties import Properties
from src.pynumic.properties.loss import total_loss


class Propagation(Properties):
    """Propagation."""

    def calc_neurons(self) -> None:
        """Calculating neurons."""
        length, dec = self.len_input, 0
        for i in range(len(self.neurons)):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j in range(len(self.neurons[i])):
                self.neurons[i][j].value = num = 0.0
                for k in range(len(self.data_weight[i][j])):
                    if k < length:
                        if i > 0:
                            self.neurons[i][j].value += self.neurons[dec][k].value * self.data_weight[i][j][k]
                        else:
                            self.neurons[i][j].value += self.data_input[k] * self.data_weight[i][j][k]
                    else:
                        self.neurons[i][j].value += self.data_weight[i][j][k]
                    num += 1

                if self.activation_mode == self.LINEAR:
                    if num > 0:
                        self.neurons[i][j].value /= num
                else:
                    self.neurons[i][j].value = self.get_activation(self.neurons[i][j].value)

    @total_loss(0)  # self.loss_mode
    def calc_loss(self) -> Iterable[float]:
        """Calculating and return the total error of the output neurons."""
        for i in range(self.len_output):
            yield self.data_target[i] - self.neurons[self.last_layer_ind][i].value

    # def calc_loss(self) -> float:
    #     """Calculating and return the total error of the output neurons.
    #     """
    #     # TODO: try-catch
    #     error = 0.0
    #     for i in range(self.len_output):
    #         self.neurons[self.last_layer_ind][i].miss = (
    #                 self.data_target[i] - self.neurons[self.last_layer_ind][i].value
    #         )
    #         match self.loss_mode:
    #             case loss.LossMode.MSE | loss.LossMode.RMSE:
    #                 error += self.neurons[self.last_layer_ind][i].miss ** 2
    #             case loss.LossMode.ARCTAN:
    #                 error += math.atan(self.neurons[self.last_layer_ind][i].miss) ** 2
    #             case loss.LossMode.AVG:
    #                 error += math.fabs(self.neurons[self.last_layer_ind][i].miss)
    #
    #     error /= self.len_output
    #     if self.loss_mode == loss.LossMode.RMSE:
    #         error = math.sqrt(error)
    #
    #     match True:
    #         case math.isnan(error):
    #             logging.log(0, "perceptron.calc_loss: loss not-a-number value")
    #         case math.isinf(error):
    #             logging.log(0, "perceptron.calc_loss: loss is infinity")
    #
    #     return error

    def calc_miss(self) -> None:
        """Calculating the error of neuron in hidden layers."""
        if self.last_layer_ind > 0:
            for i in range(self.last_layer_ind - 1, -1, -1):
                inc = i + 1
                for j in range(len(self.neurons[i])):
                    self.neurons[i][j].miss = 0.0
                    for k in range(len(self.neurons[inc])):
                        self.neurons[i][j].miss += self.neurons[inc][k].miss * self.data_weight[inc][k][j]

    def update_weights(self) -> None:
        """Update weights."""
        length, dec = self.len_input, 0
        for i in range(len(self.data_weight)):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j in range(len(self.data_weight[i])):
                grad = self.rate * self.neurons[i][j].miss * self.get_derivative(self.neurons[i][j].value)

                for k in range(len(self.data_weight[i][j])):
                    if k < length:
                        value = self.neurons[dec][k].value
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
