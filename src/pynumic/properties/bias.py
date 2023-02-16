class Bias:
    """Bias.

    The neuron bias, false or true (required field for a config).
    """

    _bias: bool = True

    @property
    def bias(self) -> bool:
        return self._bias

    @bias.setter
    def bias(self, value: bool) -> None:
        self._bias = value
