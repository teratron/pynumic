from typing import Any

from .architecture.architecture import architecture
# from pynumic.properties.activation import ActivationMode
# from pynumic.properties.loss import LossMode
# from pynumic.interface import Interface
# from pynumic.architecture.perceptron.perceptron import Perceptron
from .architecture.perceptron.perceptron import Perceptron


class Pynumic:
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

    # def _init__(self, reader: str = "", **props: Any) -> None:
    #     pass

    def __new__(cls, reader: str = "", **props: Any) -> Perceptron:
        print("__new__", cls)
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
        inst = architecture(reader, **props)
        print("inst", type(inst))
        return inst
        # return architecture(reader, **props)
        # return super().__new__(architecture(reader, **props))
