"""Interface for neural network."""

from src.pynumic.propagation import Propagation


class Interface(Propagation):
    """Interface for neural network."""

    # all = ["verify", "query", "train", "and_train", "write"]

    MAX_ITERATION: int = 1e+06

    #
    # var GetMaxIteration = getMaxIteration

    def verify(self, input_data: list[float], target_data: list[float]) -> float:
        """Verifying dataset."""
        return verify(self, input_data, target_data)

    def query(self, input_data: list[float]) -> list[float]:
        """Querying dataset."""
        return query(self, input_data)

    def train(self, input_data: list[float], target_data: list[float]) -> tuple[int, float]:
        """Training dataset."""
        return train(self, input_data, target_data)

    def and_train(self, target_data: list[float]) -> tuple[int, float]:
        """Training dataset after the query."""
        return and_train(self, target_data)

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


def verify(obj: Interface, input_data: list[float], target_data: list[float]) -> float:
    print(obj, input_data, target_data)
    return 0.0


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


def query(obj: Interface, input_data: list[float]) -> list[float]:
    """TODO:"""
    if len(input_data) > 0:
        if obj.is_init and obj.len_input == len(input_data):
            pass

    print("query***:", obj, input_data)
    # obj.calc_neurons()
    return [0, 1]


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


def train(obj: Interface, input_data: list[float], target_data: list[float]) -> tuple[int, float]:
    """TODO:"""
    print(obj, input_data, target_data)
    return 0, 0.1


def and_train(obj: Interface, target_data: list[float]) -> tuple[int, float]:
    """TODO:"""
    print(obj, target_data)
    return 0, 0.1

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