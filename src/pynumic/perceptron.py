import random
from typing import Any

from src.pynumic.interface.interface import Interface
from src.pynumic.properties.properties import Properties

from src.pynumic.interface.initialize import initialize
from src.pynumic.interface.query import query
from src.pynumic.interface.set_props import set_props
from src.pynumic.interface.train import and_train, train
from src.pynumic.interface.verify import verify
from src.pynumic.propagation import Propagation


class Perceptron(Interface, Propagation):  # Interface Pynumic
    """Perceptron is neural network."""

    name: str = "perceptron"
    type: str = "Perceptron"
    description: str = __doc__
    print("Perceptron")

    def __init__(self, **props: Any) -> None:
        # print("__init__", props)
        # self._props = props
        # if "name" in props:
        #     del props["name"]

        # Weights
        if "weights" in props:
            self.weights = props["weights"]
            del props["weights"]
        else:
            self.weights = [
                [[random.uniform(-0.5, 0.5) for _ in range(5)]
                 for _ in range(5)]
                for _ in range(5)
            ]

        # Config
        if "config" in props:
            self.config = props["config"]
            del props["config"]

        Properties.__init__(self, **props)

    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        initialize(self, *args, **kwargs)

    def set_props(self, *args: Any, **kwargs: Any) -> None:
        set_props(self, *args, **kwargs)

    def verify(self, *args: Any, **kwargs: Any) -> float:
        return verify(self, *args, **kwargs)

    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        return query(self, args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        return train(self, *args, **kwargs)

    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        return and_train(self, *args, **kwargs)
