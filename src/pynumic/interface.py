"""TODO:"""
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import overload, Any

import matplotlib.pyplot as plt

from pynumic.propagation import Propagation
from pynumic.properties import WeightsType


# from threading import Lock

@dataclass()
class DataPlot:
    """DataPlot."""

    iter: list[int]
    loss: list[float]
    avg: list[float]


class Interface(Propagation):
    """Interface for neural network."""

    MAX_ITERATION: int = 1_000  # _000
    """Maximum number of iterations after which training is forcibly terminated."""

    __slots__ = (
        "_config",
        "__weights",
        "__is_init",
        "__is_query"
        # "__mutex"
    )

    _config: str | None
    __weights: WeightsType
    # __mutex: Lock

    def __init__(self, **props: Any) -> None:
        super().__init__(**props)
        self.__is_init = False
        self.__is_query = False
        # self.__mutex = Lock()
        self.data_plot = DataPlot([], [])

    def verify(self, data_input: list[float], data_target: list[float]) -> float:
        """Verifying dataset."""
        if not self.__is_init:
            if self._init(len(data_input), len(data_target)):
                self.__is_init = True
        # self.__mutex.acquire()
        # self.__mutex.release()

        # self.__mutex.acquire()
        # try:
        #     """... доступ к общим ресурсам"""
        # finally:
        #     self.__mutex.release()  # освобождаем блокировку независимо от результата
        self._data_input = data_input
        self._data_target = data_target
        self._calc_neurons()
        return self._calc_loss()

    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""
        # if not self.__is_init:
        #     raise ValueError(f"{__name__}: not initialized")
        if not self.__is_init:
            if self._init():
                self.__is_init = True
        self._data_input = data_input
        self._calc_neurons()
        self.__is_query = True
        return [n.value for n in self._neurons[self._last_ind]]

    def train(self, data_input: list[float], data_target: list[float]) -> tuple[int, float]:
        """Training dataset."""
        if not self.__is_init:
            if self._init(len(data_input), len(data_target)):
                self.__is_init = True
        self._data_input = data_input
        self._data_target = data_target
        return self.__train()

    def and_train(self, data_target: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""
        if not self.__is_init:
            raise ValueError(f"{__name__}: not initialized")
        self._data_target = data_target
        return self.__train()

    def __train(self) -> tuple[int, float]:
        # max_loss = 0.0
        min_loss = 1.0
        min_count = 0
        prev_loss = 0
        # prev_loss = (0, 0)
        # prev_ratio = 0
        for count in range(1, self.MAX_ITERATION):
            if not self.__is_query:
                self._calc_neurons()
            else:
                self.__is_query = False

            loss = self._calc_loss()
            # if loss > max_loss:
            #     max_loss = loss

            # print(f"+++ {count = }, {loss = :.10f}")
            self.data_plot.iter.append(count)
            self.data_plot.loss.append(loss)
            # self.data_plot.loss.append(round(loss, 10))
            # print(f"+++ {self.data_plot.iter[count-1] = }, {self.data_plot.loss[count-1] = :.8f}")
            print(f"+++ {loss}  {self.data_plot.loss[count - 1]:.38f}")

            if loss < min_loss:
                min_loss = loss
                min_count = count
                self.__weights = deepcopy(self._weights)
                # print(f"--------- {count}, {loss:.33f}, {loss.as_integer_ratio()}")  #
                # if loss < self._loss_limit:
                #     self._weights = deepcopy(self.__weights)
                #     return min_count, min_loss

            # if count % 10000 == 0:
            #     # print(f"+++ {count}, {loss:.33f}, {str(loss)[str(loss).rfind('e-') + 2:]}, {(loss - prev_loss):.33f}")
            #     print(f"+++ {count}, {loss:.33f}, {loss - prev_loss}")
            # prev_loss = loss

            # ratio = loss.as_integer_ratio()[0]
            # if prev_ratio == ratio and count % 10000 == 0:
            #     print(f"******** {prev_ratio} - {ratio}")
            # prev_ratio = ratio

            self._calc_miss()
            self._update_weights()

        if min_count > 0:
            self._weights = deepcopy(self.__weights)

        fig, ax = plt.subplots()
        # print(self.data_plot.iter, self.data_plot.loss)
        ax.plot(self.data_plot.iter[:min_count + 10], self.data_plot.loss[:min_count + 10])
        ax.set(
            xlabel='iter',
            ylabel='loss',
            title='Loss'
        )
        ax.grid()
        # fig.savefig("test.png")
        plt.show()

        return min_count, min_loss

    @overload
    def write(self, filename: str | None = None, *, flag: str | None = None) -> None:
        """
        Writes configuration and weights to one file:
        - write("perceptron.json")

        Writes configuration only:
        - write("perceptron.json", flag="config")

        Writes only weights:
        - write("perceptron.json", flag="weights")

        :param filename:
        :param flag:
        :return: None
        """
        ...

    @overload
    def write(self, *, config: str | None = None, weights: str | None = None) -> None:
        """
        Writes configuration only:
        - write(config="perceptron_config.json")

        Writes only weights:
        - write(weights="perceptron_weights.json")

        Writes 2 files, configuration separately and weights separately:
        - write(config="perceptron_config.json", weights="perceptron_weights.json")

        Writes configuration and weights to one file:
        - write(config="perceptron.json", weights="perceptron.json")

        :param config:
        :param weights:
        :return: None
        """
        ...

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
                    print("Некорректный флаг или флаг отсутствует, будет записана конфигурация и веса")
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
