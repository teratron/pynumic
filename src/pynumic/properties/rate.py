class Rate:
    """Rate.

    Learning coefficient (greater than 0.0 and less than or equal to 1.0).
    """

    DEFAULT_RATE: float = 0.3

    _rate: float = DEFAULT_RATE

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate = Rate.__check(value)

    @classmethod
    def __check(cls, value: float) -> float:
        return cls.DEFAULT_RATE if 0 >= value > 1 else value
