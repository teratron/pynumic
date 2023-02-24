"""PyNumic."""
import json
import os
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
        - Pynumic(**{"bias": True, "rate": 0.3})
        - Pynumic("config/perceptron.json")
        - Pynumic("config/perceptron.json", bias=True, rate=0.3)
        - Pynumic("{'bias': true, 'rate': 0.5}")
        - Pynumic("{'bias': true, 'rate': 0.5}, bias=True, rate=0.3")
    """

    name: str = "pynumic"
    type: str = "Pynumic"
    config: str | None = None
    mutex: Lock | None = None
    is_init: bool = False

    def __init__(self, reader: str = "", **props: Any) -> None:
        """Returns a new neural network instance of one of the architectures.
        :param reader: string variable through which is passed:
                * Filename of json config ("config.json")
                * Directly json dump passed as a string ("{'bias': true, ...}")
        :param props: properties of the neural network.
        :type reader: str
        :type props: dict[str, Any]
        :return:
        :rtype:
        """
        if reader != "":
            props = _get_props_from(reader)

            if "config" in props:
                self.config = props["config"]
                del props["config"]

        super().__init__()
        Properties.__init__(self, **props)

    def __call__(self, reader: str = "", **props: Any) -> None:
        self.__init__(reader, **props)

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
    try:
        data: dict[str, Any]
        if os.path.isfile(reader):
            filename = os.path.normpath(reader)
            _, extension = os.path.splitext(filename)

            if extension == ".json":
                with open(filename, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data.update(config=filename)
            else:
                raise FileExistsError(f"{__name__}: incorrect config file extension: {extension}")
        else:
            data = json.loads(reader)

        return data
    except json.JSONDecodeError as err:
        print(f"{__name__}: JSONDecodeError: {err}")
        raise
