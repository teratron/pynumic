from typing import Any

from pynumic.architecture.architecture import architecture
from pynumic.architecture.perceptron.perceptron import Perceptron


# from pynumic.architecture.architecture import architecture, NNN
# from pynumic.properties.activation import ActivationMode
# from pynumic.properties.loss import LossMode


class Pynumic:
    """Access point to neural network.

    Pynumic(reader: str, **props)

    Examples:

        - Pynumic()
        - Pynumic("perceptron")
        - Pynumic("perceptron", bias=True, rate=0.3)
        - Pynumic(name="perceptron", bias=True, rate=0.3)
        - Pynumic("config/perceptron.json")
        - Pynumic("{'name': 'perceptron', 'bias': true, 'rate': 0.3}")
        - Pynumic(**{"name": "perceptron", "bias": True, "rate": 0.3})

    Keyword arguments:

    reader -- string variable through which is passed:
        - Name of the neural network ("perceptron" or "hopfield");
        - Filename of json config ("config.json");
        - Directly json dump passed as a string ("{'name': 'perceptron', ...}").
    **kwargs -- properties of the neural network.
    """

    def __new__(cls, reader: str = "", **props: Any) -> Perceptron:
        return architecture(reader, **props)
