"""PyNumic."""
import json
import os
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
        """Returns a new neural network instance of one of the architectures.
        :param reader: string variable through which is passed:
                * Name of the neural network ("perceptron" or "hopfield")
                * Filename of json config ("config.json")
                * Directly json dump passed as a string ("{'name': 'perceptron', ...}")
        :param props: properties of the neural network.
        :type reader:
        :type props:
        :return:
        :rtype:
        """
        if reader != "":
            props = _get_props_from(reader)

        # if props != {}:
        #     if "name" in props:
        #         return architecture(props["name"], **props)
        #     else:
        #         raise NameError(f"{__name__}: missing field: name")

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

        super().__init__()
        Properties.__init__(self, **props)
        print(self.__dict__, self.__dir__())

    def __call__(self, **props: Any) -> None:
        """Set properties of neural network."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.__str__()}: {self.__dict__}"

    def __dir__(self) -> list[str]:
        """Returns all members and all public methods."""
        return (
                ["__class__", "__doc__", "__module__"]
                + [m for cls in self.__class__.mro() for m in cls.__dict__ if m[0] != "_"]
                + [m for m in self.__dict__ if m[0] != "_"]
        )


def _get_props_from(reader: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    if os.path.isfile(reader):
        filename = os.path.normpath(reader)
        _, extension = os.path.splitext(filename)

        if extension == ".json":
            with open(filename, "r", encoding="utf-8") as handle:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError as err:
                    print(f"{__name__}: JSONDecodeError: {err}")

            data.update(config=filename)
            print(data)
        else:
            raise FileExistsError(
                    f"{__name__}: incorrect config file extension: {extension}"
            )
    else:
        try:
            data = json.loads(reader)
        except json.JSONDecodeError as err:
            print(f"{__name__}: JSONDecodeError: {err}")

    return data
