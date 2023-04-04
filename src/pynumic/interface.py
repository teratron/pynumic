"""TODO: Interface for neural network."""
import json
from copy import deepcopy
from typing import overload, Any

from pynumic.propagation import Propagation
from pynumic.properties import WeightsType


class Interface(Propagation):
    """Interface for neural network."""

    MAX_ITERATION: int = 1_000_000
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
        max_loss = 0.0
        min_loss = 1.0
        min_count = 0
        prev_loss = 0.0
        for count in range(1, self.MAX_ITERATION):
            if not self.__is_query:
                self._calc_neurons()
            else:
                self.__is_query = False

            loss = self._calc_loss()
            if loss > max_loss:
                max_loss = loss

            if loss < min_loss:
                min_loss = loss
                min_count = count
                self.__weights = deepcopy(self._weights)
                print("---------", count, loss)
                if loss < self._loss_limit:
                    self._weights = deepcopy(self.__weights)
                    return min_count, min_loss

            if count % 10000 == 0:
                print(f"+++, {count}, {loss:.36f}")  # , self._neurons[0][0], self._weights[0][0][0]

            if round(loss, 31) == prev_loss and count % 10000 == 0:
                print(count, loss)  # , self._neurons[0][0], self._weights[0][0][0]
            prev_loss = round(loss, 31)

            self._calc_miss()
            self._update_weights()

        if min_count > 0:
            self._weights = deepcopy(self.__weights)

        return min_count, min_loss

    @overload
    def write(self, filename: str | None = None, *, flag: str | None = None) -> None:
        ...

    @overload
    def write(self, *, config: str | None = None, weights: str | None = None) -> None:
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
        print(self.__dict__)

        props: dict[str, Any] = {
            key.lstrip("_"): value for key, value in self.__dict__.items() if key != "_weights"
        }
        props.update({"weights": self.__dict__.get("_weights")})

        with open("linear.json", "w", newline="\n", encoding="utf-8") as handle:
            json.dump(props, handle, indent="\t")
        # if filename is None and flag is None and config is None and weights is None:
        #     if self._config:
        #         with open(self._config, "w", newline="\n", encoding="utf-8") as handle:
        #             json.dump(self._weights, handle, skipkeys=True, indent="\t")

        # if filename is None:
        #     filename = self._config
        # else:
        #     if os.path.isfile(filename):
        #         filename = os.path.normpath(filename)
        #
        # with open(filename, "w", newline="\n", encoding="utf-8") as handle:
        #     json.dump(self._weights, handle, skipkeys=True, indent="\t")
