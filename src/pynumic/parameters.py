class Parameters:
    """Parameters."""

    _len_input: int = 0
    _len_output: int = 0
    _last_ind: int = 0
    _prev_ind: int = 0

    # def __init__(self) -> None:
    #     self._last_ind = length - 1
    #     self._prev_ind = self._last_ind - 1

    @property
    def last_ind(self) -> int:
        return self._last_ind

    @last_ind.setter
    def last_ind(self, value: int) -> None:
        self._last_ind = value
        self._prev_ind = self._last_ind - 1
