"""TODO:"""
import json
import os
from abc import ABC, abstractmethod
from typing import overload, Any

from pynumic.properties import WeightsType


class Interface(ABC):
    """Interface for neural network."""

    def __init__(self, config: str, weights: WeightsType) -> None:
        self.__config = config
        self.__weights = weights

    @abstractmethod
    def verify(self, data_input: list[float], data_target: list[float]) -> float:
        """Verifying dataset."""

    @abstractmethod
    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""

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
            flag="config
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

    # @abstractmethod
    def write(
            self,
            filename: str | None = None,
            *,
            flag: str | None = None,
            config: str | None = None,
            weights: str | None = None
    ) -> None:
        """Writes the configuration and/or weights to a file.

        Writes configuration and weights to one file:

        - write()
        - write("perceptron.json")
        - write(config="perceptron.json", weights="perceptron.json")

        Writes configuration only:

        - write(config="perceptron.json")
        - write("perceptron.json", flag="config")

        Writes only weights:

        - write(weights="perceptron_weights.json")
        - write("perceptron.json", flag="weights")

        Writes 2 files, configuration separately and weights separately:

        - write(config="perceptron.json", weights="perceptron_weights.json")
        """
        props: dict[str, Any] = {
            key.lstrip("_"): value for key, value in self.__dict__.items() if key != "_weights"
        }
        # props.update({"weights": self.__dict__.get("_weights")})

        if filename is None and flag is None and config is None and weights is None:
            if self.__config:
                props.update({"weights": self.__dict__.get("_weights")})
                with open(self.__config, "w", newline="\n", encoding="utf-8") as handle:
                    json.dump(props, handle, indent="\t")
            else:
                print("Отсутствует файл или путь к файлу конфигурации")

        if filename and os.path.normpath(filename):
            match flag:
                case "config":
                    with open(filename, "w", newline="\n", encoding="utf-8") as handle:
                        json.dump(props, handle, indent="\t")
                case "weight" | "weights":
                    with open(filename, "w", newline="\n", encoding="utf-8") as handle:
                        json.dump({"weights": self.__weights}, handle, indent="\t")
                case None | _:
                    print(
                        "Некорректный флаг или флаг отсутствует, будет записана конфигурация и веса"
                    )
                    props.update({"weights": self.__dict__.get("_weights")})
                    with open(filename, "w", newline="\n", encoding="utf-8") as handle:
                        json.dump(props, handle, indent="\t")

        if config and weights:
            props.update({"weights": self.__dict__.get("_weights")})
            with open(config, "w", newline="\n", encoding="utf-8") as handle:
                json.dump(props, handle, indent="\t")
        else:
            if config and os.path.normpath(config):
                with open(config, "w", newline="\n", encoding="utf-8") as handle:
                    json.dump(props, handle, indent="\t")

            if weights and os.path.normpath(weights):
                with open(weights, "w", newline="\n", encoding="utf-8") as handle:
                    json.dump({"weights": self.__weights}, handle, indent="\t")
