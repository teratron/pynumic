"""TODO: Initialization."""
import random
from dataclasses import dataclass

from pynumic.properties import Properties, WeightsType


@dataclass
class Neuron:
    """Neuron."""

    value: float
    miss: float


class Initialization(Properties):
    """Initialization neural network."""

    # Neurons
    neurons: list[list[Neuron]]

    # Transfer data
    data_weight: WeightsType
    data_input: list[float]
    data_target: list[float]
    data_output: list[float]

    # Settings
    len_input: int = 0
    len_output: int = 0
    last_layer_ind: int = 0
    is_init: bool = False

    def init_from_new(self, len_input: int, len_target: int) -> bool:
        """TODO:"""
        self.len_input = len_input
        self.len_output = len_target
        self.last_layer_ind = len(self._hidden_layers)

        if self.last_layer_ind > 0 and self._hidden_layers[0] == 0:
            self.last_layer_ind = 0

        layer: list[int] = self._hidden_layers.copy()
        if self.last_layer_ind > 0:
            layer.append(self.len_output)
        else:
            layer = [self.len_output]

        len_layer = len(layer)

        bias_input = self.len_input + int(self._bias)
        biasLayer: int

        self.weights = make(pkg.Float3Type, len_layer)
        self.data_weight = make(pkg.Float3Type, len_layer)
        self.neurons = make([][] * Neuron, len_layer)
        for i, v := range layer:
            self.weights[i] = make(pkg.Float2Type, v)
            self.data_weight[i] = make(pkg.Float2Type, v)
            self.neurons[i] = make([] * Neuron, v)
            if i > 0:
                biasLayer = int(layer[i - 1]) + int(self._bias)

            for j := 0; j < int(v); j++:
                if i > 0:
                    self.weights[i][j] = make(pkg.Float1Type, biasLayer)
                    self.data_weight[i][j] = make(pkg.Float1Type, biasLayer)
                else:
                    self.weights[i][j] = make(pkg.Float1Type, bias_input)
                    self.data_weight[i][j] = make(pkg.Float1Type, bias_input)

                for k := range self.weights[i][j]:
                    if self.activation_mode == self.LINEAR:
                        self.weights[i][j][k] = .5
                    else:
                        self.weights[i][j][k] = params.GetRandFloat()

                self.neurons[i][j] = & Neuron
                {}

        self.weights = [
            [
                [
                    random.uniform(-0.5, 0.5) for _ in range(5)
                ] for _ in range(5)
            ] for _ in range(5)
        ]

        return self._init_completion()

    def init_from_weight(self) -> bool:
        """TODO:"""
        length = len(self.weights)
        self.last_layer_ind = length - 1
        self.len_output = len(self.weights[self.last_layer_ind])
        self.len_input = len(self.weights[0][0])

        if length > 1 and len(self.weights[0]) + 1 == len(self.weights[1][0]):
            self._bias = True
            self.len_input -= 1

        if self.last_layer_ind > 0:
            self._hidden_layers = [len(self.weights[i]) for i in enumerate(self._hidden_layers)]
        else:
            self._hidden_layers = [0]

        self.data_weight = [[[0.0 for _ in w] for w in v] for v in self.weights]
        self.neurons = [[Neuron(0, 0) for _ in v] for v in self.weights]

        return self._init_completion()

    def _init_completion(self) -> bool:
        self.data_input = [0.0 for _ in range(self.len_input)]
        self.data_target = [0.0 for _ in range(self.len_output)]
        self.data_output = self.data_target.copy()
        self.is_init = True

        return self.is_init

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
