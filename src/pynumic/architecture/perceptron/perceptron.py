import random
from typing import Any

from pynumic.architecture.perceptron.interface.initialize import initialize
from pynumic.architecture.perceptron.interface.query import query
from pynumic.architecture.perceptron.interface.set_props import set_props
from pynumic.architecture.perceptron.interface.train import train, and_train
from pynumic.architecture.perceptron.interface.verify import verify
from pynumic.architecture.perceptron.propagation import Propagation
from pynumic.architecture.perceptron.properties import Properties
from pynumic.interface import Interface


class Perceptron(Interface, Propagation):  # Properties
    """Perceptron is neural network.
    """

    print("Perceptron")
    name: str = "perceptron"
    type: str = "Perceptron"
    description: str = __doc__

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
                [[random.uniform(-0.5, 0.5) for _ in range(5)] for _ in range(5)]
                for _ in range(5)
            ]

        # Config
        if "config" in props:
            self.config = props["config"]
            del props["config"]

        Properties.__init__(self, **props)
        # Propagation.__init__(self, self)

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
