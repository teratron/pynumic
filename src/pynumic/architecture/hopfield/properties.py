class Properties:
    """Properties of neural network."""

    # __slots__ = (
    #     "_name",
    #     "_energy",
    # )
    print("Hopfield props")

    def __init__(self, name: str, *, energy: float = 0.3) -> None:
        self._name: str = name
        self._energy: float = energy

    @property
    def energy(self) -> float:
        """Energy."""
        return self._energy

    @energy.setter
    def energy(self, energy: float):
        self._energy = self.check_energy(energy)

    @staticmethod
    def check_energy(energy: float) -> float:
        return 0.3 if energy <= 0 or energy > 1 else energy
