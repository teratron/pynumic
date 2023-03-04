"""TODO: Interface for neural network."""
from copy import deepcopy
from threading import Lock

from src.pynumic.propagation import Propagation


class Interface(Propagation):
    """Interface for neural network."""

    MAX_ITERATION: int = 1_000_000_000
    """Maximum number of iterations after which training is forcibly terminated."""

    mutex: Lock = Lock()
    buff_input: list[float]
    is_query: bool = False

    def verify(
            self, data_input: list[float], data_target: list[float]
    ) -> float:
        """Verifying dataset."""
        if not self.is_init:
            self.init_from_new(len(data_input), len(data_target))  # TODO:

        # self.mutex.acquire()
        # self.mutex.release()

        # self.mutex.acquire()
        # try:
        #     """... доступ к общим ресурсам"""
        # finally:
        #     self.mutex.release()  # освобождаем блокировку независимо от результата

        self.calc_neurons(data_input)

        return self.calc_loss(data_target)

    def query(self, data_input: list[float]) -> list[float]:
        """Querying dataset."""
        if not self.is_init:
            raise ValueError(f"{__name__}: not initialized")

        self.calc_neurons(data_input)
        self.buff_input = data_input
        self.is_query = True

        return [n.value for n in self.neurons[self.layers["last_index"]]]

    def train(
            self, data_input: list[float], data_target: list[float]
    ) -> tuple[int, float]:
        """Training dataset."""
        if not self.is_init:
            self.init_from_new(len(data_input), len(data_target))  # TODO:

        return self.__train(data_input, data_target)

    def and_train(self, data_target: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""
        if not self.is_init:
            raise ValueError(f"{__name__}: not initialized")

        return self.__train(self.buff_input, data_target)

    def __train(
            self, data_input: list[float], data_target: list[float]
    ) -> tuple[int, float]:
        min_loss = 1.0
        min_count = 0
        for count in range(1, self.MAX_ITERATION):
            if not self.is_query:
                self.calc_neurons(data_input)
            else:
                self.is_query = False

            loss = self.calc_loss(data_target)
            if loss < min_loss:
                min_loss = loss
                min_count = count
                self.data_weight = deepcopy(self.weights)
                if loss < self._loss_limit:
                    self.weights = deepcopy(self.data_weight)
                    return min_count, min_loss

            self.calc_miss()
            self.update_weights(data_input)

        if min_count > 0:
            self.weights = deepcopy(self.data_weight)

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

# def _verify(obj: Interface, input_data: list[float], target_data: list[float]) -> float:
#     print(obj, input_data, target_data)
#     return 0.0


# // Verify verifying dataset.
# func (nn *NN) Verify(input []float64, target ...[]float64) float64 {
# 	var err error
# 	if len(input) > 0 {
# 		if len(target) > 0 && len(target[0]) > 0 {
# 			nn.mutex.Lock()
# 			defer nn.mutex.Unlock()
#
# 			if !nn.isInit {
# 				nn.Init(len(input), len(target[0]))
# 			} else {
# 				if nn.lenInput != len(input) {
# 					err = fmt.Errorf("invalid number of elements in the input data")
# 					goto ERROR
# 				}
# 				if nn.lenOutput != len(target[0]) {
# 					err = fmt.Errorf("invalid number of elements in the target data")
# 					goto ERROR
# 				}
# 			}
#
# 			if nn.Weights[0][0][0] != 0 {
# 				nn.weights = nn.Weights
# 			}
#
# 			nn.input = pkg.ToFloat1Type(input)
# 			nn.target = pkg.ToFloat1Type(target[0])
#
# 			nn.calcNeuron()
# 			return nn.calcLoss()
# 		} else {
# 			err = pkg.ErrNoTarget
# 		}
# 	} else {
# 		err = pkg.ErrNoInput
# 	}
#
# ERROR:
# 	log.Printf("perceptron.NN.Verify: %v\n", err)
# 	return -1
# }


# def _query(obj: Interface, input_data: list[float]) -> list[float]:
#     # if len(input_arg) > 0:
#     #     if obj.is_init and obj.len_input == len(input_arg):
#     #         pass
#
#     if obj.weights is not None:
#         obj.data_weight = obj.weights.copy()
#
#     obj.data_input = input_data
#     obj.calc_neurons()
#
#     obj.data_output = obj.neurons[obj.last_layer_ind]
#
#     return obj.data_output

# print("query***:", obj, input_arg)
# obj.calc_neurons()
# return [0, 1]


# // Query querying dataset.
# func (nn *NN) Query(input []float64) []float64 {
# 	var err error
# 	if len(input) > 0 {
# 		nn.mutex.Lock()
# 		defer nn.mutex.Unlock()
#
# 		if !nn.isInit {
# 			err = pkg.ErrInit
# 			goto ERROR
# 		} else if nn.lenInput != len(input) {
# 			err = fmt.Errorf("invalid number of elements in the input data")
# 			goto ERROR
# 		}
#
# 		if nn.Weight[0][0][0] != 0 {
# 			nn.weight = nn.Weight
# 		}
#
# 		nn.input = pkg.ToFloat1Type(input)
#
# 		nn.calcNeuron()
# 		for i, n := range nn.neuron[nn.lastLayerIndex] {
# 			nn.output[i] = float64(n.value)
# 		}
# 		return nn.output
# 	} else {
# 		err = pkg.ErrNoInput
# 	}
#
# ERROR:
# 	log.Printf("perceptron.NN.Query: %v\n", err)
# 	return nil
# }


# def train(obj: Interface, input_data: list[float], target_data: list[float]) -> tuple[int, float]:
#     print(obj, input_data, target_data)
#     return 0, 0.1
#
#
# def and_train(obj: Interface, target_data: list[float]) -> tuple[int, float]:
#     print(obj, target_data)
#     return 0, 0.1

# // MaxIteration the maximum number of iterations after which training is forcibly terminated.
# const MaxIteration int = 1e+06
#
# var GetMaxIteration = getMaxIteration
#
# func getMaxIteration() int {
# 	return MaxIteration
# }
#
# // Train training dataset.
# func (nn *NN) Train(input []float64, target ...[]float64) (count int, loss float64) {
# 	var err error
# 	if len(input) > 0 {
# 		if len(target) > 0 && len(target[0]) > 0 {
# 			nn.mutex.Lock()
# 			defer nn.mutex.Unlock()
#
# 			if !nn.isInit {
# 				nn.Init(len(input), len(target[0]))
# 			} else {
# 				if nn.lenInput != len(input) {
# 					err = fmt.Errorf("invalid number of elements in the input data")
# 					goto ERROR
# 				}
# 				if nn.lenOutput != len(target[0]) {
# 					err = fmt.Errorf("invalid number of elements in the target data")
# 					goto ERROR
# 				}
# 			}
#
# 			if nn.Weights[0][0][0] != 0 {
# 				nn.weights = nn.Weights
# 			}
#
# 			nn.input = pkg.ToFloat1Type(input)
# 			nn.target = pkg.ToFloat1Type(target[0])
#
# 			minLoss := 1.
# 			minCount := 0
# 			for count < GetMaxIteration() {
# 				count++
# 				nn.calcNeuron()
#
# 				if loss = nn.calcLoss(); loss < minLoss {
# 					minLoss = loss
# 					minCount = count
# 					nn.Weights = nn.weights
# 					if loss < nn.LossLimit {
# 						return minCount, minLoss
# 					}
# 				}
# 				nn.calcMiss()
# 				nn.updateWeight()
# 			}
# 			return minCount, minLoss
# 		} else {
# 			err = pkg.ErrNoTarget
# 		}
# 	} else {
# 		err = pkg.ErrNoInput
# 	}
#
# ERROR:
# 	log.Printf("perceptron.NN.Train: %v\n", err)
# 	return 0, -1
# }
#
# // AndTrain the training dataset after the query.
# func (nn *NN) AndTrain(target []float64) (count int, loss float64) {
# 	var err error
# 	if len(target) > 0 {
# 		nn.mutex.Lock()
# 		defer nn.mutex.Unlock()
#
# 		if !nn.isInit {
# 			err = pkg.ErrInit
# 			goto ERROR
# 		} else if nn.lenOutput != len(target) {
# 			err = fmt.Errorf("invalid number of elements in the target data")
# 			goto ERROR
# 		}
#
# 		if nn.Weights[0][0][0] != 0 {
# 			nn.weights = nn.Weights
# 		}
#
# 		nn.target = pkg.ToFloat1Type(target)
#
# 		isStart := true
# 		minLoss := 1.
# 		minCount := 0
# 		for count < GetMaxIteration() {
# 			count++
# 			if !isStart {
# 				nn.calcNeuron()
# 			} else {
# 				isStart = false
# 			}
#
# 			if loss = nn.calcLoss(); loss < minLoss {
# 				minLoss = loss
# 				minCount = count
# 				nn.Weights = nn.weights
# 				if loss < nn.LossLimit {
# 					return minCount, minLoss
# 				}
# 			}
# 			nn.calcMiss()
# 			nn.updateWeight()
# 		}
# 		return minCount, minLoss
# 	} else {
# 		err = pkg.ErrNoTarget
# 	}
#
# ERROR:
# 	log.Printf("perceptron.NN.AndTrain: %v\n", err)
# 	return 0, -1
# }
