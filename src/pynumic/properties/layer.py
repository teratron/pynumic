from typing import Optional

LayerType = Optional[list[int]]


class Layer:
    """Layer.
    """

    _hidden_layers: LayerType = None

    @property
    def hidden_layers(self) -> Optional[list[int]]:
        """List of the number of neuron in each hidden layers."""
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value: list[int]) -> None:
        self._hidden_layers = Layer.__check(value)

    @staticmethod
    def __check(value: Optional[list[int]]) -> list[int]:
        return [0] if value is None else value
