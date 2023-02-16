from abc import ABC, abstractmethod
from asyncio import Lock
from typing import Any, Optional


class Interface(ABC):
    """Interface for neural network.
    """

    name: str = "interface"
    type: str = "Interface"
    description: str = __doc__

    is_init: bool = False
    config: Optional[str] = None
    mutex: Lock

    @abstractmethod
    def _initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize neural network."""
        ...

    @abstractmethod
    def set_props(self, *args: Any, **kwargs: Any) -> None:
        """Set properties of neural network."""
        ...

    @abstractmethod
    def verify(self, *args: Any, **kwargs: Any) -> float:
        """Verifying dataset."""
        ...

    @abstractmethod
    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        """Querying dataset."""
        ...

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset."""
        ...

    @abstractmethod
    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset after the query."""
        ...

    def write(
            self,
            *,
            filename: str | None = None,
            flag: str | None = None,
            config: str | None = None,
            weights: str | None = None
    ) -> None:
        """Writes the configuration and weights to a file.

        * Writes configuration and weights to one file:
        write("perceptron.json")
        write(config="perceptron.json", weights="perceptron.json")

        * Writes configuration only:
        write(config="perceptron.json")
        write("perceptron.json", flag="config")

        * Writes only weights:
        write(weights="perceptron_weights.json")
        write("perceptron.json", flag="weights")

        * Writes 2 files, configuration separately and weights separately:
        write(config="perceptron.json", weights="perceptron_weights.json")
        """
        ...

    def __str__(self) -> str:
        return "%s.%s" % (self.__class__.__name__, self.name)

    def __repr__(self) -> str:
        return "<%s: %r>" % (self.__str__(), self.__dict__)

    def __dir__(self) -> list[str]:
        """Returns all members and all public methods."""
        return (
                ["__class__", "__doc__", "__module__"] +
                [m for cls in self.__class__.mro() for m in cls.__dict__ if m[0] != "_"] +
                [m for m in self.__dict__ if m[0] != "_"]
        )
