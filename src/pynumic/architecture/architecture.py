import json
import os
from typing import Any

from src.pynumic.architecture.hopfield.hopfield import Hopfield
from src.pynumic.architecture.perceptron.perceptron import Perceptron


# NN = Perceptron | Hopfield
# from pynumic.interface import Interface


def architecture(reader: str, **props: Any) -> Perceptron | Hopfield:  #
    """Returns an instance of one of the architectures.
    :param reader:
    :param props:
    :type reader:
    :type props:
    :return:
    :rtype:
    """

    name = reader.lower()

    if name == "perceptron":
        from src.pynumic.architecture.perceptron.perceptron import Perceptron
        return Perceptron(**props)
    elif name == "hopfield":
        from src.pynumic.architecture.hopfield.hopfield import Hopfield
        return Hopfield(**props)
    else:
        if reader != "":
            props = _get_props_from(reader)

        if props != {}:
            if "name" in props:
                return architecture(props["name"], **props)
            else:
                raise NameError(f"{__name__}: missing field: name")

    return architecture("perceptron", **props)


def _get_props_from(reader: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    if os.path.isfile(reader):
        filename = os.path.normpath(reader)
        _, extension = os.path.splitext(filename)

        if extension == ".json":
            with open(filename) as handle:
                data = json.load(handle)
            data.update(config=filename)
            print(data)
        else:
            raise FileExistsError(f"{__name__}: incorrect config file extension: {extension}")
    else:
        try:
            data = json.loads(reader)
        except json.JSONDecodeError as err:
            print(f"{__name__}: JSONDecodeError: {err}")

    return data
