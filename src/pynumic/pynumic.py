from typing import Any

from pynumic.architecture.architecture import architecture
from pynumic.architecture.hopfield.hopfield import Hopfield
from pynumic.architecture.perceptron.perceptron import Perceptron


class Pynumic(Perceptron, Hopfield):
    """Access point to neural network.

    Pynumic(reader: str, **props)

    :Examples:
        - Pynumic()
        - Pynumic("perceptron")
        - Pynumic("perceptron", bias=True, rate=0.3)
        - Pynumic(name="perceptron", bias=True, rate=0.3)
        - Pynumic("config/perceptron.json")
        - Pynumic("{'name': 'perceptron', 'bias': true, 'rate': 0.3}")
        - Pynumic(**{"name": "perceptron", "bias": True, "rate": 0.3})
    """

    # __instance = None

    def __new__(cls, reader: str = "", **props: Any):
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
        # cls.__instance = architecture(reader, **props)
        # if cls.__instance is None:
        #     cls.__instance = super().__new__(cls)
        # return cls.__instance
        return architecture(reader, **props)
