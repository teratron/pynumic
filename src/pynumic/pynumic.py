"""PyNumic."""

from asyncio import Lock
from typing import Any

from pynumic.interface import Interface
from pynumic.properties import Properties
from pynumic.properties.layers import LayersType


class Pynumic(Interface):
    """Access point to neural network.

    Pynumic(reader: str, **props)

    :Examples:
        - Pynumic()
        - Pynumic(bias=True, rate=0.3)
        - Pynumic("config/perceptron.json")
        - Pynumic("{'bias': true, 'rate': 0.3}")
        - Pynumic(**{"bias": True, "rate": 0.3})
    """

    # __slots__ = (
    #     "_bias",
    #     "_hidden_layers",
    #     "_activation_mode",
    #     "_loss_mode",
    #     "_loss_limit",
    #     "_rate"
    # )

    name: str = "pynumic"
    type: str = "Pynumic"
    is_init: bool = False
    config: str | None = None
    mutex: Lock | None = None

    def __init__(
            self,
            *,
            bias: bool = True,
            hidden_layers: LayersType = None,
            activation_mode: int = Properties.TANH,
            loss_mode: int = Properties.RMSE,
            loss_limit: float = Properties.DEFAULT_LOSS_LIMIT,
            rate: float = Properties.DEFAULT_RATE,
    ) -> None:
        self.bias: bool = bias
        self.hidden_layers: LayersType = hidden_layers
        self.activation_mode: int = activation_mode
        self.loss_mode: int = loss_mode
        self.loss_limit: float = loss_limit
        self.rate: float = rate
        print(self.__dict__, self.__dir__())

        super().__init__()
        Properties.__init__(self, **self.__dict__)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Set properties of neural network."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.__str__()}: {self.__dict__}"

    def __dir__(self) -> list[str]:
        """Returns all members and all public methods."""
        return (
                ["__class__", "__doc__", "__module__"]
                + [m for cls in self.__class__.mro()
                   for m in cls.__dict__ if m[0] != "_"]
                + [m for m in self.__dict__ if m[0] != "_"]
        )
