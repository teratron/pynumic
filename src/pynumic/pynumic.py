"""PyNumic."""
import random
from asyncio import Lock
from typing import Any

from pynumic.interface import Interface
from pynumic.properties import Properties


class Pynumic(Interface):
    """Access point to neural network.

    Pynumic(reader: str, **props)

    :Examples:
        - Pynumic()
        - Pynumic(bias=True, rate=0.3)
        - Pynumic("config/perceptron.json")
        - Pynumic("{'bias': true, 'rate': 0.3}")
        - Pynumic(**{"bias": True, "rate": 0.3})
    """

    name: str = "pynumic"
    type: str = "Pynumic"
    config: str | None = None
    mutex: Lock | None = None
    is_init: bool = False

    def __init__(self, reader: str = "", **props: Any) -> None:
        self.reader = reader

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

        super().__init__()
        Properties.__init__(self, **props)
        print(self.__dict__, self.__dir__())

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Set properties of neural network."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.__str__()}: {self.__dict__}"

    def __dir__(self) -> list[str]:
        """Returns all members and all public methods."""
        return (
                ["__class__", "__doc__", "__module__"]
                + [m for cls in self.__class__.mro()
                   for m in cls.__dict__ if m[0] != "_"]
                + [m for m in self.__dict__ if m[0] != "_"]
        )
