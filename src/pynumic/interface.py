"""TODO: Interface for neural network."""
from copy import deepcopy

from pynumic.propagation import Propagation
from pynumic.properties import WeightsType


# from threading import Lock


class Interface(Propagation):
    """Interface for neural network."""

    MAX_ITERATION: int = 1_000_000_000
    """Maximum number of iterations after which training is forcibly terminated."""

    # __mutex: Lock = Lock()
    __weights: WeightsType
    __is_init: bool = False
    __is_query: bool = False

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

        return [n.value for n in self.neurons[self._last_ind]]

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
        for count in range(1, self.MAX_ITERATION):
            if not self.__is_query:
                self._calc_neurons()
            else:
                self.__is_query = False

            loss = self._calc_loss()
            # if loss > max_loss:
            #     max_loss = loss

            if loss < min_loss:
                min_loss = loss
                min_count = count
                self.__weights = deepcopy(self._weights)
                # print("---------", count, loss)
                if loss < self._loss_limit:
                    self._weights = deepcopy(self.__weights)
                    return min_count, min_loss

            # if count % 10000 == 0:
            #     print(count, loss, self.neurons[0][0], self._weights[0][0][0])

            self._calc_miss()
            self._update_weights()

        if min_count > 0:
            self._weights = deepcopy(self.__weights)

        return min_count, min_loss

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

# // WriteConfig writes the configuration and weights to the Filer interface object.
# func (nn *NN) WriteConfig(name ...string) (err error) {
# 	if len(name) > 0 {
# 		switch d := utils.GetFileType(name[0]).(type) {
# 		case error:
# 			err = d
# 		case utils.Filer:
# 			err = d.Encode(nn)
# 		}
# 	} else if nn.config != nil {
# 		err = nn.config.Encode(nn)
# 	} else {
# 		err = pkg.ErrNoArgs
# 	}
#
# 	if err != nil {
# 		err = fmt.Errorf("perceptron.NN.WriteConfig: %w", err)
# 		log.Print(err)
# 	}
# 	return
# }
#
# // WriteWeights writes weights to the Filer interface object.
# func (nn *NN) WriteWeights(name string) (err error) {
# 	switch d := utils.GetFileType(name).(type) {
# 	case error:
# 		err = d
# 	case utils.Filer:
# 		err = d.Encode(nn.Weights)
# 	}
#
# 	if err != nil {
# 		err = fmt.Errorf("perceptron.NN.WriteWeights: %w", err)
# 		log.Print(err)
# 	}
# 	return
# }
