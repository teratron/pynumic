from typing import Any

from pynumic.architecture.hopfield.properties import Properties
from pynumic.interface import Interface


class Hopfield(Interface, Properties):
    """Hopfield is neural network.
    """

    print("Hopfield")
    name: str = "hopfield"
    type: str = "Hopfield"
    description: str = __doc__

    def __init__(self, **props) -> None:
        print("Hopfield init")
        # Properties.__init__(self, self.name, **props)
        Properties.__init__(self, **props)

    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize neural network."""
        pass

    def set_props(self, *args: Any, **kwargs: Any) -> None:
        """Set properties of neural network."""
        pass

    def verify(self, *args: Any, **kwargs: Any) -> float:
        """Verifying dataset."""
        pass

    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        """Querying dataset."""
        pass

    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset."""
        pass

    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset after the query."""
        pass

    def write(self, *args: Any, **kwargs: Any) -> None:
        """Writes the configuration and weights to a file."""
        pass
