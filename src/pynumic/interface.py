"""TODO:"""
from abc import ABC, abstractmethod
from typing import Any


class Interface(ABC):
    """Interface for neural network."""

    @abstractmethod
    def verify(self, data_input: list[float], data_target: list[float]) -> float:
    #def verify(self, data_input: list[float], *args: Any) -> float:
        """Verifying dataset."""
        ...

    @abstractmethod
    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""
        ...

    @abstractmethod
    def train(self, data_input: list[float], data_target: list[float]) -> tuple[int, float]:
    #def train(self, data_input: list[float], *args: Any) -> tuple[int, float]:
        """Training dataset."""
        ...

    @abstractmethod
    def and_train(self, data_target: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""
        ...

    @abstractmethod
    def write(self, *args: Any, **kwargs: Any) -> None:
        """Writes the configuration and/or weights to a file."""
        ...
