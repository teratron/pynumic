class Parameters:
    """Parameters."""

    _last_ind: int = 0
    prev_ind: int = -1
    len_input: int = 0
    len_output: int = 0
    is_init: bool = False

    @property
    def last_ind(self) -> int:
        return self._last_ind

    @last_ind.setter
    def last_ind(self, value: int) -> None:
        self._last_ind = value
        self.prev_ind = value - 1
