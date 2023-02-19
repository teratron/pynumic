from pynumic.properties.activation import Activation
from pynumic.properties.bias import Bias
from pynumic.properties.layer import Layer, LayerType
from pynumic.properties.loss import Loss
from pynumic.properties.rate import Rate


class Neuron:
    def __init__(self, value: float, miss: float) -> None:
        self.value = value
        self.miss = miss


class Properties(Bias, Layer, Activation, Loss, Rate):
    """Properties of neural network."""

    # __slots__ = (
    #     "_name",
    #     "_bias",
    #     "_hidden_layers",
    #     "_activation_mode",
    #     "_loss_mode",
    #     "_loss_limit",
    #     "_rate"
    # )

    def __init__(
            self,
            # name: str,  # TODO: ?
            *,
            bias: bool = True,
            hidden_layers: LayerType = None,
            activation_mode: int = Activation.TANH,
            loss_mode: int = Loss.RMSE,
            loss_limit: float = 0.1e-3,
            rate: float = Rate.DEFAULT_RATE,
    ) -> None:
        # self.name: str = name
        self.bias: bool = bias
        self.hidden_layers: LayerType = hidden_layers
        self.activation_mode: int = activation_mode
        self.loss_mode: int = loss_mode
        self.loss_limit: float = loss_limit
        self.rate: float = rate

        # Layer.__init__(self, hidden_layers)
        # Activation.__init__(self, activation_mode)
        # Loss.__init__(self, loss_mode, loss_limit)
        # Rate.__init__(self, rate)
        # Bias.__init__(self, bias)

    # Neurons
    neurons: list[list[Neuron]] | None = None

    # Transfer data
    data_weight: list[list[list[float]]] | None = None
    data_input: list[float] | None = None
    data_target: list[float] | None = None
    data_output: list[float] | None = None

    # Settings
    len_input: int = 0
    len_output: int = 0
    last_layer_ind: int = 0
