import random
from asyncio import Lock
from typing import Any

from pynumic.interface.initialize import initialize
from pynumic.interface.query import query
from pynumic.interface.set_props import set_props
from pynumic.interface.train import train, and_train
from pynumic.interface.verify import verify
from pynumic.properties import Properties


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
