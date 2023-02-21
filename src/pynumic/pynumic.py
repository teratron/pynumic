"""PyNumic."""

import random
from asyncio import Lock
from typing import Any

from src.pynumic.interface.initialize import initialize
from src.pynumic.interface.query import query
from src.pynumic.interface.set_props import set_props
from src.pynumic.interface.train import and_train, train
from src.pynumic.interface.verify import verify
from src.pynumic.properties.properties import Properties


class Pynumic:
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
    description: str = __doc__

    is_init: bool = False
    config: str | None = None
    mutex: Lock | None = None

    def __init__(self, **props: Any) -> None:
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

        Properties.__init__(**props)

    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize neural network."""
        initialize(self, *args, **kwargs)

    def set_props(self, *args: Any, **kwargs: Any) -> None:
        """Set properties of neural network."""
        set_props(self, *args, **kwargs)

    def verify(self, input: list[float], target: list[float]) -> float:
        """Verifying dataset."""
        return verify(self, input, target)

    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        """Querying dataset."""
        return query(self, args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset."""
        return train(self, *args, **kwargs)

    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset after the query."""
        return and_train(self, *args, **kwargs)

    def write(
            self,
            *,
            filename: str | None = None,
            flag: str | None = None,
            config: str | None = None,
            weights: str | None = None,
    ) -> None:
        """Writes the configuration and weights to a file.

        * Writes configuration and weights to one file:
        write("perceptron.json")
        write(config="perceptron.json", weights="perceptron.json")

        * Writes configuration only:
        write(config="perceptron.json")
        write("perceptron.json", flag="config")

        * Writes only weights:
        write(weights="perceptron_weights.json")
        write("perceptron.json", flag="weights")

        * Writes 2 files, configuration separately and weights separately:
        write(config="perceptron.json", weights="perceptron_weights.json")
        """
        pass

    def __str__(self) -> str:
        return "%s.%s" % (self.__class__.__name__, self.name)

    def __repr__(self) -> str:
        return "<%s: %r>" % (self.__str__(), self.__dict__)

    def __dir__(self) -> list[str]:
        """Returns all members and all public methods."""
        return (
                ["__class__", "__doc__", "__module__"]
                + [m for cls in self.__class__.mro()
                   for m in cls.__dict__ if m[0] != "_"]
                + [m for m in self.__dict__ if m[0] != "_"]
        )

    # def __new__(cls, reader: str = "", **props: Any) -> Perceptron:
    #     """Returns a new neural network instance of one of the architectures.
    #     :param reader: string variable through which is passed:
    #             * Filename of json config ("config.json")
    #             * Directly json dump passed as a string ("{'name': 'perceptron', ...}")
    #     :param props: properties of the neural network.
    #     :type reader:
    #     :type props:
    #     :return:
    #     :rtype:
    #     """
    #     # return super().__new__(architecture(reader, **props))
    #     return architecture(reader, **props)
