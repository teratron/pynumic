"""Interface for neural network."""

from typing import Any

from pynumic.interface.query import query
from pynumic.interface.train import and_train, train
from pynumic.interface.verify import verify
from pynumic.propagation import Propagation


class Interface(Propagation):
    """Interface for neural network."""

    # all = ["verify", "query", "train", "and_train", "write"]

    def verify(self, input_data: list[float], target_data: list[float]) -> float:
        """Verifying dataset."""
        return verify(self, input_data, target_data)

    def query(self, *args: Any, **kwargs: Any) -> list[float]:
        """Querying dataset."""
        return query(self, args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset."""
        return train(self, *args, **kwargs)

    def and_train(self, *args: Any, **kwargs: Any) -> tuple[int, float]:
        """Training dataset after the query."""
        return and_train(self, *args, **kwargs)

    def write(
            self,
            *,
            filename: str | None = None,
            flag: str | None = None,
            config: str | None = None,
            weights: str | None = None,
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
        pass
