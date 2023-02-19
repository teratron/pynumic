from typing import Any

from pynumic.architecture.hopfield.properties import Properties
from pynumic.interface.interface import Interface


# from pynumic.pynumic import Pynumic


class Hopfield(Interface, Properties):  # Interface
    """Hopfield is neural network."""

    print("Hopfield")
    name: str = "hopfield"
    type: str = "Hopfield"
    description: str = __doc__

    def __init__(self, **props) -> None:
        print("Hopfield init")
        # Properties.__init__(self, self.name, **props)
        Properties.__init__(self, **props)

    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        # initialize(self, *args, **kwargs)
        pass

    def set_props(self, *args: Any, **kwargs: Any) -> None:
        # set_props(self, *args, **kwargs)
        pass

    def verify(self, *args: Any, **kwargs: Any) -> float:
        # return verify(self, *args, **kwargs)
        return 0

    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        # return query(self, args, **kwargs)
        return [0]

    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        # return train(self, *args, **kwargs)
        return 0, 0

    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        # return and_train(self, *args, **kwargs)
        return 0, 0
