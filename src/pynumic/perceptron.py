"""TODO:"""
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pynumic.interface import Interface
from pynumic.propagation import Propagation
from pynumic.properties import WeightsType, Neuron


@dataclass()
class DataArray:
    """TODO:"""

    count: list[int]
    loss: list[float]
    # resistance: list[float]


class Perceptron(Propagation, Interface):
    """Interface for neural network."""

    MAX_ITERATION: int = 1_000_000
    """Maximum number of iterations after which training is forcibly terminated."""

    __slots__ = ("_config", "__weights", "__is_query", "data")

    __weights: WeightsType

    def __init__(self, **props: Any) -> None:
        super().__init__(**props)
        self.__is_query = False
        self._config: str | None = None
        self.data: DataArray = DataArray([], [])

    def __init(self, len_input: int, len_target: int) -> bool:
        self._params.len_input = len_input
        self._params.len_output = len_target
        self._params.layers = [self._params.len_output]
        weights: list[int] = [self._params.len_input + int(self._bias)]

        if self._hidden_layers:
            self._params.last_ind = len(self._hidden_layers)
            self._params.layers = self._hidden_layers + self._params.layers
            weights += list(map(lambda x: x + int(self._bias), self._hidden_layers))

        self._weights = [
            [
                [
                    -1.0  # 0.555
                    if self._activation_mode == self.LINEAR
                    else round(random.uniform(-0.5, 0.5), 3)
                    for _ in range(weights[i])
                ]
                for _ in range(v)
            ]
            for i, v in enumerate(self._params.layers)
        ]

        print(self._params.layers)

        self._neurons = [[Neuron(0, 0) for _ in range(v)] for v in self._params.layers]
        self._params.is_init = True
        del weights

        return self._params.is_init

    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""
        if not self._params.is_init:
            raise ValueError(f"{__name__}: not initialized")

        self._data_input = data_input
        self._calculate_neurons()
        # self._params.is_query = True
        self.__is_query = True

        return [n.value for n in self._neurons[self._params.last_ind]]

    def verify(self, data_input: list[float], data_target: list[float]) -> float:
        """Verifying dataset."""
        if not self._params.is_init:
            if self.__init(len(data_input), len(data_target)):
                self._params.is_init = True

        self._data_input = data_input
        self._data_target = data_target
        self._calculate_neurons()

        # noinspection PyArgumentList
        return self._calculate_loss()

    def train(self, data_input: list[float], data_target: list[float]) -> tuple[int, float]:
        """Training dataset."""
        if not self._params.is_init:
            if not self.__init(len(data_input), len(data_target)):
                raise ValueError(f"{__name__}: not initialized")

        self._data_input = data_input
        self._data_target = data_target

        return self.__train()

    def and_train(self, data_target: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""
        if not self._params.is_init:
            raise ValueError(f"{__name__}: not initialized")
        self._data_target = data_target
        return self.__train()

    def __train(self) -> tuple[int, float]:
        min_loss = 1.0
        min_count = 0

        # prev_loss = 0
        # prev_loss = (0, 0)
        # prev_ratio = 0
        for count in range(1, self.MAX_ITERATION):
            if not self.__is_query:
                # if not self._params.is_query:
                self._calculate_neurons()
            else:
                # self._params.is_query = False
                self.__is_query = False

            # noinspection PyArgumentList
            loss = self._calculate_loss()

            self.data.count.append(count)
            self.data.loss.append(loss)

            if loss < min_loss:
                min_loss = loss
                min_count = count
                self.__weights = deepcopy(self._weights)
                # print(f"--------- {count}, {loss:.33f}, {loss.as_integer_ratio()}")  #
                if loss < self._loss_limit:
                    self._weights = deepcopy(self.__weights)
                    return min_count, min_loss

            # if count % 10000 == 0:
            #     # print(f"+++ {count}, {loss:.33f}, {str(loss)[str(loss).rfind('e-') + 2:]}, {(loss - prev_loss):.33f}")
            #     print(f"+++ {count}, {loss:.33f}, {loss - prev_loss}")
            # prev_loss = loss

            # ratio = loss.as_integer_ratio()[0]
            # if prev_ratio == ratio and count % 10000 == 0:
            #     print(f"******** {prev_ratio} - {ratio}")
            # prev_ratio = ratio

            self._calculate_miss()
            self._update_weights()

        if min_count > 0:
            self._weights = deepcopy(self.__weights)

        return min_count, min_loss

    def write(
            self,
            filename: str | None = None,
            *,
            flag: str | None = None,
            config: str | None = None,
            weights: str | None = None,
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
            if self._config:
                props.update({"weights": self.__dict__.get("_weights")})
                with open(self._config, "w", newline="\n", encoding="utf-8") as handle:
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
                        json.dump({"weights": self._weights}, handle, indent="\t")
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
                    json.dump({"weights": self._weights}, handle, indent="\t")
