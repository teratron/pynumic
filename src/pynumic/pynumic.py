"""TODO:"""
import json
import os
from typing import Any

from pynumic.perceptron import Perceptron
from pynumic.properties import Properties


class Pynumic(Perceptron):
    """Access point to neural network.

    Pynumic(reader: str, **props: Any)

    Examples:
        - Pynumic()
        - Pynumic(bias=True, rate=0.3)
        - Pynumic(**{"bias": True, "rate": 0.3})
        - Pynumic("config/perceptron.json")
        - Pynumic("config/perceptron.json", bias=True, rate=0.3)
        - Pynumic("{'bias': true, 'rate': 0.5}")
        - Pynumic("{'bias': true, 'rate': 0.5}, bias=True, rate=0.3")

    pn = Pynumic():
        - pn() -- reset to default
        - pn(bias=True, rate=0.3)
        - pn(**{"bias": True, "rate": 0.3})
        - pn("config/perceptron.json")
        - pn("config/perceptron.json", bias=True, rate=0.3)
        - pn("{'bias': true, 'rate': 0.5}")
        - pn("{'bias': true, 'rate': 0.5}, bias=True, rate=0.3")
    """

    name: str = "pynumic"

    def __init__(self, reader: str = "", **props: Any) -> None:
        """Returns a new neural network instance of one of the architectures.
        :param reader: string variable through which is passed:
                - Filename of json config ("config.json")
                - Directly json dump passed as a string ("{'bias': true, ...}")
        :param props: properties of the neural network.
        :type reader: str
        :type props: dict[str, Any]
        :return:
        :rtype:
        """
        # super().__init__(**self.__get_props(reader, **props))
        Properties.__init__(self, **self.__get_props(reader, **props))

    def __call__(self, reader: str = "", **props: Any) -> None:
        props = self.__get_props(reader, **props)
        if props:
            for key in props.keys():
                if key in self.__dir__():
                    setattr(self, key, props[key])

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.__str__()}: {self.__dict__}"

    def __dir__(self) -> list[str]:
        return (
            ["__class__", "__doc__", "__module__"]
            + [m for cls in self.__class__.mro() for m in cls.__dict__ if m[0] != "_"]
            + [m for m in self.__dict__ if m[0] != "_"]
        )

    def __get_props(self, reader: str, **props: Any) -> dict[str, Any]:
        if reader != "":
            kwargs = _get_props_from(reader)
            if "config" in kwargs:
                self._config = kwargs["config"]
                del kwargs["config"]
            props |= kwargs

        return props


def _get_props_from(reader: str) -> dict[str, Any]:
    try:
        data: dict[str, Any]
        if os.path.isfile(reader):
            filename = os.path.normpath(reader)
            _, ext = os.path.splitext(filename)
            if ext == ".json":
                with open(filename, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data.update(config=filename)
            else:
                raise FileExistsError(
                    f"{__name__}: incorrect config file extension: {ext}"
                )
        else:
            data = json.loads(reader)
        return data
    except json.JSONDecodeError as err:
        print(f"{__name__}: JSONDecodeError: {err}")
        raise
