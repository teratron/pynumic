from src.pynumic.properties.energy import Energy


class Properties(Energy):
    """Properties of neural network."""

    # __slots__ = (
    #     "_name",
    #     "_energy",
    # )
    print("Hopfield props")

    def __init__(
            self,
            *,
            # name: str,
            energy: float = 0.3
    ) -> None:
        # self._name: str = name
        self._energy: float = energy
