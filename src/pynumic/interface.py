"""TODO: interface.py - """
from abc import ABC, abstractmethod
from typing import overload


class Interface(ABC):
    """Interface for neural network."""

    @abstractmethod
    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""

    @abstractmethod
    def verify(self, data_input: list[float], data_target: list[float]) -> float:
        """Verifying dataset."""

    @abstractmethod
    def train(self, data_input: list[float], data_target: list[float]) -> tuple[int, float]:
        """Training dataset."""

    @abstractmethod
    def and_train(self, data_target: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""

    @overload
    def write(self, filename: str | None = None, *, flag: str | None = None) -> None:
        """Writes configuration and weights to one file:
        - write("perceptron.json")

        Writes configuration only:
        - write(
            "perceptron.json",
            flag="config"
        )

        Writes only weights:
        - write(
            "perceptron.json",
            flag="weights"
        )

        :param filename:
        :param flag:
        - config
        - weights
        :return: None
        """

    @overload
    def write(self, *, config: str | None = None, weights: str | None = None) -> None:
        """Writes configuration only:
        - write(
            config="perceptron_config.json"
        )

        Writes only weights:
        - write(
            weights="perceptron_weights.json"
        )

        Writes 2 files, configuration separately and weights separately:
        - write(
            config="perceptron_config.json",
            weights="perceptron_weights.json"
        )

        Writes configuration and weights to one file:
        - write(
            config="perceptron.json",
            weights="perceptron.json"
        )

        :param config:
        :param weights:
        :return: None
        """

    @abstractmethod
    def write(
        self,
        filename: str | None = None,
        *,
        flag: str | None = None,
        config: str | None = None,
        weights: str | None = None,
    ) -> None:
        """Writes the configuration and/or weights to a file."""
