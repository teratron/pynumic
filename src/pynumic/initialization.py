"""TODO: Initialization."""
import random
from dataclasses import dataclass

from src.pynumic.properties import Properties, WeightsType


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


class Initialization(Properties):
    """Initialization neural network."""

    neurons: list[list[Neuron]]
    data_weight: WeightsType
    # layers: list[int]
    layers: dict[str, int] = {
        "len_input":  0,
        "len_output": 0,
        "last_index": 0,
        "prev_index": 0,
    }
    # len_input: int = 0
    # len_output: int = 0
    # last_layer_ind: int = 0
    is_init: bool = False

    def init_from_new(self, len_input: int, len_target: int) -> None:
        """TODO:"""
        # self.layers = [len_input + int(self._bias)]
        # self.layers.extend(self._hidden_layers)
        # self.layers.append(len_target)

        self.layers = {
            "len_input":  len_input,
            "len_output": len_target,
            "last_index": len(self._hidden_layers),
        }
        # self.layers["len_input"] = len_input
        # self.layers["len_output"] = len_target
        # self.layers["last_index"] = len(self._hidden_layers)

        # self.len_input = len_input
        # self.len_output = len_target
        # self.last_layer_ind = len(self._hidden_layers)

        if self.layers["last_index"] > 0 and self._hidden_layers[0] == 0:
            self.layers["last_index"] = 0

        self.layers["prev_index"] = self.layers["last_index"] - 1

        layers: list[int] = self._hidden_layers.copy()
        if self.layers["last_index"] > 0:
            layers.append(len_target)
        else:
            layers = [len_target]

        len_layers = len(layers)
        bias_input = len_input + int(self._bias)
        bias_layer: int = 0  # TODO:

        # self.weights = make(pkg.Float3Type, len_layers)
        # self.data_weight = make(pkg.Float3Type, len_layers)
        # self.neurons = make([][]*Neuron, len_layers)
        self.neurons = [[Neuron(0, 0) for _ in range(v)] for v in layers]

        for i, v in enumerate(layers):
            # self.weights[i] = make(pkg.Float2Type, v)
            # self.data_weight[i] = make(pkg.Float2Type, v)
            # self.neurons[i] = make([] * Neuron, v)
            if i > 0:
                bias_layer = int(layers[i - 1]) + int(self._bias)

            # for j in range(v):
            #     if i > 0:
            #         self.weights[i][j] = make(pkg.Float1Type, bias_layer)
            #         self.data_weight[i][j] = make(pkg.Float1Type, bias_layer)
            #     else:
            #         self.weights[i][j] = make(pkg.Float1Type, bias_input)
            #         self.data_weight[i][j] = make(pkg.Float1Type, bias_input)
            #
            #     for k in self.weights[i][j]:
            #         if self.activation_mode == self.LINEAR:
            #             self.weights[i][j][k] = 0.5
            #         else:
            #             self.weights[i][j][k] = random.uniform(-0.5, 0.5)  # params.GetRandFloat()

            # self.neurons[i][j] = &Neuron{}

        self.weights = [
            [[random.uniform(-0.5, 0.5) for _ in range(5)] for _ in range(5)]
            for _ in range(len_layers)
        ]
        self.is_init = True

    def init_from_weight(self) -> None:
        """TODO:"""
        length = len(self.weights)
        self.layers["last_index"] = length - 1
        # self.len_output = len(self.weights[self.last_layer_ind])
        # self.len_input = len(self.weights[0][0])

        self.layers = {
            "len_input":  len(self.weights[0][0]),
            "len_output": len(self.weights[self.layers["last_index"]]),
        }

        if length > 1 and len(self.weights[0]) + 1 == len(self.weights[1][0]):
            self._bias = True
            self.layers["len_input"] -= 1

        if self.layers["last_index"] > 0:
            self._hidden_layers = [
                len(self.weights[i]) for i, _ in enumerate(self._hidden_layers)
            ]
        else:
            self._hidden_layers = [0]

        self.data_weight = [[[0 for _ in w] for w in v] for v in self.weights]
        self.neurons = [[Neuron(0, 0) for _ in v] for v in self.weights]
        self.is_init = True

# if __name__ == "__main__":
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
# 	nn.weights = make(pkg.Float3Type, lenLayer)
# 	nn.neuron = make([][]*neuron, lenLayer)
# 	for i, v := range layer {
# 		nn.Weights[i] = make(pkg.Float2Type, v)
# 		nn.weights[i] = make(pkg.Float2Type, v)
# 		nn.neuron[i] = make([]*neuron, v)
# 		if i > 0 {
# 			biasLayer = int(layer[i-1]) + bias
# 		}
#
# 		for j := 0; j < int(v); j++ {
# 			if i > 0 {
# 				nn.Weights[i][j] = make(pkg.Float1Type, biasLayer)
# 				nn.weights[i][j] = make(pkg.Float1Type, biasLayer)
# 			} else {
# 				nn.Weights[i][j] = make(pkg.Float1Type, biasInput)
# 				nn.weights[i][j] = make(pkg.Float1Type, biasInput)
# 			}
# 			for k := range nn.weights[i][j] {
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
# 	nn.weights = make(pkg.Float3Type, length)
# 	nn.neuron = make([][]*neuron, length)
# 	for i, v := range nn.Weights {
# 		length = len(v)
# 		nn.weights[i] = make(pkg.Float2Type, length)
# 		nn.neuron[i] = make([]*neuron, length)
# 		for j, w := range v {
# 			nn.weights[i][j] = make(pkg.Float1Type, len(w))
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


# def _init_from_new(len_input: int, len_target: int) -> None:
#     _hidden_layers = [42, 21]
#     _bias: bool = False
#     layers = [len_input + int(_bias)]
#     layers.extend(_hidden_layers)
#     layers.append(len_target)
#     print(layers)
#
#
# if __name__ == "__main__":
#     _init_from_new(3, 2)
