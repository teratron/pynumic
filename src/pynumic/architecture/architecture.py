import json
import os
from typing import Any


def architecture(reader: str, **props: Any) -> Any:
    """Returns an instance of one of the architectures.
    :param reader:
    :type reader:
    :param props:
    :type props:
    :return:
    """

    if reader.lower() == "perceptron":
        from pynumic.architecture.perceptron.perceptron import Perceptron

        return Perceptron(**props)
    elif reader.lower() == "hopfield":
        from pynumic.architecture.hopfield.hopfield import Hopfield

        return Hopfield(**props)
    else:
        if reader != "":
            props = _get_props_from(reader)

        if props != {}:
            if "name" in props:
                return architecture(props["name"], **props)
            else:
                raise NameError("missing field: name")

    # return Perceptron()
    return None


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
            raise FileExistsError("incorrect config file extension: " + extension)
    else:
        try:
            data = json.loads(reader)
        except json.JSONDecodeError as err:
            print("JSONDecodeError", err)

    return data
