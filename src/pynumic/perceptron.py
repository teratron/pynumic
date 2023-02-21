import random
from typing import Any

from pynumic.interface.initialize import initialize
from pynumic.interface.query import query
from pynumic.interface.set_props import set_props
from pynumic.interface.train import train, and_train
from pynumic.interface.verify import verify
from pynumic.propagation import Propagation
from pynumic.properties import Properties
from src.pynumic.interface import Interface


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
