"""Rate."""


class Rate:
    """Learning coefficient (greater than 0.0 and less than or equal to 1.0)."""

    DEFAULT_RATE: float = 0.3
    _rate: float = DEFAULT_RATE

    @property
    def rate(self) -> float:
        """TODO:"""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate = Rate.check(value)

    @classmethod
    def check(cls, value: float) -> float:
        """Checking whether the value corresponds to normal conditions."""
        return cls.DEFAULT_RATE if value <= 0 or value > 1 else value
