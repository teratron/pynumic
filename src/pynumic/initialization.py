"""TODO: Initialization."""
import random
from dataclasses import dataclass

from pynumic.properties import Properties  # , WeightsType


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


class Initialization(Properties):
    """initialization neural network."""

    neurons: list[list[Neuron]]
    # __weights: WeightsType
    _len_input: int = 0
    _len_output: int = 0
    _last_ind: int = 0
    _prev_ind: int = 0

    def _init(self, len_input: int = 0, len_target: int = 0) -> bool:
        is_init: bool = False
        if self._weights:
            is_init = self.__init_from_weight()
        elif len_input > 0 and len_target > 0:
            is_init = self.__init_from_new(len_input, len_target)

        return is_init

    def __init_from_new(self, len_input: int, len_target: int) -> bool:
        self._len_input = len_input
        self._len_output = len_target

        weights: list[int] = [self._len_input + int(self._bias)]
        layers: list[int] = [self._len_output]
        if self._hidden_layers:
            self._last_ind = len(self._hidden_layers)
            weights += list(map(lambda x: x + int(self._bias), self._hidden_layers))
            layers = self._hidden_layers + layers

        self._prev_ind = self._last_ind - 1
        self._weights = [
            [
                [
                    0.5 if self._activation_mode == self.LINEAR
                    else random.uniform(-0.5, 0.5)
                    for _ in range(weights[i])
                ] for _ in range(v)
            ] for i, v in enumerate(layers)
        ]
        self.neurons = [[Neuron(0, 0) for _ in range(v)] for v in layers]

        # print(self._len_input, self._len_output)
        # print(self._prev_ind, self._last_ind)
        # print(self._weights)
        # print(self.neurons)
        del weights, layers

        return True

    def __init_from_weight(self) -> bool:
        length = len(self._weights)
        self._last_ind = length - 1
        self._prev_ind = self._last_ind - 1
        self._len_input = len(self._weights[0][0])
        self._len_output = len(self._weights[self._last_ind])

        if length > 1 and len(self._weights[0]) + 1 == len(self._weights[1][0]):
            self._bias = True
            self._len_input -= 1

        if self._last_ind > 0:
            self._hidden_layers = [
                len(self._weights[i]) for i, _ in enumerate(self._hidden_layers)
            ]

        self.neurons = [[Neuron(0, 0) for _ in v] for v in self._weights]
        # self.__weights = [[[0 for _ in w] for w in v] for v in self._weights]

        return True

# [
#     [
#         [0.5, 0.5, 0.5, 0.5],
#         [0.5, 0.5, 0.5, 0.5],
#         [0.5, 0.5, 0.5, 0.5]
#     ],
#     [
#         [0.5, 0.5, 0.5, 0.5],
#         [0.5, 0.5, 0.5, 0.5]
#     ],
#     [
#         [0.5, 0.5, 0.5],
#         [0.5, 0.5, 0.5]
#     ]
# ]

# if __name__ == "__main__":
#     mode = 2
#     _weights = [
#         [
#             [
#                 random.uniform(-0.5, 0.5)
#                 if mode == 0
#                 else 0.5 for _ in range([3, 4][i])
#             ] for _ in range(v)
#         ] for i, v in enumerate([3, 1])
#     ]
#     print(_weights)
#     for i, num in enumerate([1, 3, 7]):
#         print(i, num)
#     bias = True
#     print(list(map(lambda x: x + int(bias), [1, 3, 7])))
# l = [3]
# i = [2, 5, 6]
# # i.reverse()
# # l.extend(i)
# # l.reverse()
#
# #l = [x for x in i]
# l = i + l
# print(l)

#     inz = Initialization()
#     print(inz.__dict__)

# // Init initialize.
# func (nn *NN) Init(data ...interface{}) {
# 	var err error
# 	if len(data) > 0 {
# 		switch value := data[0].(type) {
# 		case utils.Filer:
# 			if _, ok := value.(utils.FileError); !ok {
# 				if len(nn.Weights) > 0 {
# 					nn.initFromWeight()
# 				}
# 				nn.config = value
# 			}
# 		case int:
# 			if len(data) == 2 {
# 				if v, ok := data[1].(int); ok {
# 					nn.initFromNew(value, v)
# 				}
# 			}
# 		default:
# 			err = fmt.Errorf("%T %w: %v", value, pkg.ErrMissingType, value)
# 		}
# 		if err == nil {
# 			nn.initCompletion()
# 		}
# 	} else {
# 		err = pkg.ErrNoArgs
# 	}
#
# 	if err != nil {
# 		log.Printf("perceptron.NN.Init: %v\n", err)
# 	}
# }
#
# // initFromNew initialize.
# func (nn *NN) initFromNew(lenInput, lenTarget int) {
# 	nn.lenInput = lenInput
# 	nn.lenOutput = lenTarget
# 	nn.lastLayerIndex = len(nn.HiddenLayer)
# 	if nn.lastLayerIndex > 0 && nn.HiddenLayer[0] == 0 {
# 		nn.lastLayerIndex = 0
# 	}
#
# 	var layer []uint
# 	if nn.lastLayerIndex > 0 {
# 		layer = append(nn.HiddenLayer, uint(nn.lenOutput))
# 	} else {
# 		layer = []uint{uint(nn.lenOutput)}
# 	}
# 	lenLayer := len(layer)
#
# 	bias := 0
# 	if nn.Bias {
# 		bias = 1
# 	}
# 	biasInput := nn.lenInput + bias
# 	var biasLayer int
#
# 	nn.Weights = make(pkg.Float3Type, lenLayer)
# 	nn._weights = make(pkg.Float3Type, lenLayer)
# 	nn.neuron = make([][]*neuron, lenLayer)
# 	for i, v := range layer {
# 		nn.Weights[i] = make(pkg.Float2Type, v)
# 		nn._weights[i] = make(pkg.Float2Type, v)
# 		nn.neuron[i] = make([]*neuron, v)
# 		if i > 0 {
# 			biasLayer = int(layer[i-1]) + bias
# 		}
#
# 		for j := 0; j < int(v); j++ {
# 			if i > 0 {
# 				nn.Weights[i][j] = make(pkg.Float1Type, biasLayer)
# 				nn._weights[i][j] = make(pkg.Float1Type, biasLayer)
# 			} else {
# 				nn.Weights[i][j] = make(pkg.Float1Type, biasInput)
# 				nn._weights[i][j] = make(pkg.Float1Type, biasInput)
# 			}
# 			for k := range nn._weights[i][j] {
# 				if nn.ActivationMode == params.LINEAR {
# 					nn.Weights[i][j][k] = .5
# 				} else {
# 					nn.Weights[i][j][k] = params.GetRandFloat()
# 				}
# 			}
# 			nn.neuron[i][j] = &neuron{}
# 		}
# 	}
# }
#
# // initFromWeight.
# func (nn *NN) initFromWeight() {
# 	length := len(nn.Weights)
#
# 	if !nn.Bias && length > 1 && len(nn.Weights[0])+1 == len(nn.Weights[1][0]) {
# 		nn.Bias = true
# 	}
#
# 	nn.lastLayerIndex = length - 1
# 	nn.lenOutput = len(nn.Weights[nn.lastLayerIndex])
# 	nn.lenInput = len(nn.Weights[0][0])
# 	if nn.Bias {
# 		nn.lenInput -= 1
# 	}
#
# 	if nn.lastLayerIndex > 0 {
# 		nn.HiddenLayer = make([]uint, nn.lastLayerIndex)
# 		for i := range nn.HiddenLayer {
# 			nn.HiddenLayer[i] = uint(len(nn.Weights[i]))
# 		}
# 	} else {
# 		nn.HiddenLayer = []uint{0}
# 	}
#
# 	nn._weights = make(pkg.Float3Type, length)
# 	nn.neuron = make([][]*neuron, length)
# 	for i, v := range nn.Weights {
# 		length = len(v)
# 		nn._weights[i] = make(pkg.Float2Type, length)
# 		nn.neuron[i] = make([]*neuron, length)
# 		for j, w := range v {
# 			nn._weights[i][j] = make(pkg.Float1Type, len(w))
# 			nn.neuron[i][j] = &neuron{}
# 		}
# 	}
# }
#
# // initCompletion.
# func (nn *NN) initCompletion() {
# 	nn.input = make(pkg.Float1Type, nn.lenInput)
# 	nn.target = make(pkg.Float1Type, nn.lenOutput)
# 	nn.output = make([]float64, nn.lenOutput)
# 	nn.isInit = true
# }


# def _init_from_new(_len_input: int, len_target: int) -> None:
#     _hidden_layers = [42, 21]
#     _bias: bool = False
#     layers = [_len_input + int(_bias)]
#     layers.extend(_hidden_layers)
#     layers.append(len_target)
#     print(layers)
#
#
# if __name__ == "__main__":
#     _init_from_new(3, 2)
