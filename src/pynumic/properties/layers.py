"""Layers."""

LayersType = list[int] | None


class Layers:
    """Layers."""

    _hidden_layers: LayersType = None

    @property
    def hidden_layers(self) -> LayersType:
        """List of the number of neuron in each hidden layers."""
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value: list[int]) -> None:
        self._hidden_layers = Layers.check(value)

    @hidden_layers.deleter
    def hidden_layers(self) -> None:
        del self._hidden_layers

    @staticmethod
    def check(value: LayersType) -> list[int]:
        """Checking whether the value corresponds to normal conditions."""
        return [0] if value is None else value
