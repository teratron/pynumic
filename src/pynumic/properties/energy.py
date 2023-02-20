class Energy:
    """Energy."""

    DEFAULT_ENERGY: float = 0.1
    _energy: float = DEFAULT_ENERGY

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = Energy.__check(value)

    @classmethod
    def __check(cls, value: float) -> float:
        return cls.DEFAULT_ENERGY if value <= 0 or value > 1 else value
