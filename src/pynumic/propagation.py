"""TODO: Propagation"""
import math

from pynumic.initialization import Initialization


class Propagation(Initialization):
    """Propagation."""

    _data_input: list[float]
    _data_target: list[float]

    def _calc_neurons(self) -> None:
        """Calculating neurons."""
        dec, length = 0, self._len_input
        for i, layer in enumerate(self.neurons):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j, _ in enumerate(layer):
                self.__get_neuron(i, j, dec, length)

    def __get_neuron(self, i: int, j: int, dec: int, length: int) -> None:
        k = self.neurons[i][j].value = 0
        for k, weight in enumerate(self._weights[i][j]):
            if k < length:
                self.neurons[i][j].value += (
                    self.neurons[dec][k].value * weight
                    if i > 0
                    else self._data_input[k] * weight
                )
            else:
                self.neurons[i][j].value += weight

        if self._activation_mode == self.LINEAR:
            self.neurons[i][j].value /= k if k > 0 else 1
        else:
            self.neurons[i][j].value = self._get_activation(self.neurons[i][j].value)

    def _calc_loss(self) -> float:
        """Calculating and return the total error of the output neurons."""
        loss = 0.0
        for i, neuron in enumerate(self.neurons[self._last_ind]):
            neuron.miss = self._data_target[i] - neuron.value
            match self._loss_mode:
                case self.AVG:
                    loss += math.fabs(neuron.miss)
                case self.ARCTAN:
                    loss += math.atan(neuron.miss) ** 2
                case self.MSE | self.RMSE | _:
                    loss += neuron.miss ** 2

        if math.isnan(loss):
            raise ValueError(f"{__name__}: loss not-a-number value")

        if math.isinf(loss):
            raise ValueError(f"{__name__}: loss is infinity")

        loss /= self._len_output
        if self._loss_mode == self.RMSE:
            loss = math.sqrt(loss)

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

    def _update_weights(self) -> None:
        """Update weights."""
        dec, length = 0, self._len_input
        for i, weight in enumerate(self._weights):
            if i > 0:
                dec = i - 1
                length = len(self.neurons[dec])

            for j, _ in enumerate(weight):
                self.__get_weight(i, j, dec, length)

    def __get_weight(self, i: int, j: int, dec: int, length: int) -> None:
        grad = (
                self._rate
                * self.neurons[i][j].miss
                * self._get_derivative(self.neurons[i][j].value)
        )
        for k, _ in enumerate(self._weights[i][j]):
            if k < length:
                value = self.neurons[dec][k].value if i > 0 else self._data_input[k]
                if self._activation_mode == self.LINEAR:
                    self._weights[i][j][k] += grad / value if value != 0 else 0
                else:
                    self._weights[i][j][k] += grad * value
            else:
                self._weights[i][j][k] += grad
