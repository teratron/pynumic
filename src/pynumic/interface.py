from abc import ABC, abstractmethod
from typing import Any

from src.pynumic.nn import NeuralNetwork


class Interface(NeuralNetwork, ABC):
    """Interface for neural network."""

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
