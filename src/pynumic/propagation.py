"""TODO:"""
from typing import Iterable

from pynumic.loss import _total_loss
from pynumic.properties import Properties


class Propagation(Properties):
    """Propagation."""

    __slots__ = ("_data_input", "_data_target")

    _data_input: list[float]
    _data_target: list[float]

    ############################################################################
    # Forward propagation
    ############################################################################

    # Calculating neurons.
    def _calculate_neurons(self) -> None:
        dec, length = 0, self._params.len_input
        for i, layer in enumerate(self._neurons):
            if i > 0:
                dec = i - 1
                length = len(self._neurons[dec])

            for j, _ in enumerate(layer):
                self.__get_neuron(i, j, dec, length)

    def __get_neuron(self, i: int, j: int, dec: int, length: int) -> None:
        k = self._neurons[i][j].value = 0
        for k, weight in enumerate(self._weights[i][j]):
            if k < length:
                self._neurons[i][j].value += (
                    self._neurons[dec][k].value * weight if i > 0 else self._data_input[k] * weight
                )
            else:
                self._neurons[i][j].value += weight

        self._neurons[i][j].value = self._get_activation(self._neurons[i][j].value)

        if self._activation_mode == self.LINEAR:
            self._neurons[i][j].value /= k if k > 0 else 1

    # Calculating and return the total error of the output neurons.
    @_total_loss
    def _calculate_loss(self) -> Iterable[float]:
        for i in range(self._params.len_output):
            self._neurons[self._params.last_ind][i].miss = (
                self._data_target[i] - self._neurons[self._params.last_ind][i].value
            )
            yield self._neurons[self._params.last_ind][i].miss

    ############################################################################
    # Backward propagation
    ############################################################################

    # Calculating the error of neuron in hidden layers.
    def _calculate_miss(self) -> None:
        for i in range(self._params.prev_ind, -1, -1):
            inc = i + 1
            for j, _ in enumerate(self._neurons[i]):
                self._neurons[i][j].miss = 0
                for k, _ in enumerate(self._neurons[inc]):
                    self._neurons[i][j].miss += (
                        self._neurons[inc][k].miss * self._weights[inc][k][j]
                    )

    # Update weights.
    def _update_weights(self) -> None:
        dec, length = 0, self._params.len_input
        for i, weight in enumerate(self._weights):
            if i > 0:
                dec = i - 1
                length = len(self._neurons[dec])

            for j, _ in enumerate(weight):
                self.__get_weight(i, j, dec, length)

    def __get_weight(self, i: int, j: int, dec: int, length: int) -> None:
        grad = (
            self._rate * self._neurons[i][j].miss * self._get_derivative(self._neurons[i][j].value)
        )
        for k, _ in enumerate(self._weights[i][j]):
            if k < length:
                value = self._neurons[dec][k].value if i > 0 else self._data_input[k]
                if self._activation_mode == self.LINEAR:
                    self._weights[i][j][k] += grad / value if value != 0 else 0
                else:
                    self._weights[i][j][k] += grad * value
            else:
                self._weights[i][j][k] += grad
