"""TODO:"""


class Parameters:  # pylint: disable=too-few-public-methods
    """Parameters. Параметры для нормальной работы нейросети"""

    _last_ind: int = 0
    prev_ind: int = -1
    len_input: int = 0
    len_output: int = 0
    is_init: bool = False
    is_query: bool = False

    @property
    def last_ind(self) -> int:
        """TODO:"""
        return self._last_ind

    @last_ind.setter
    def last_ind(self, value: int) -> None:
        self._last_ind = value
        self.prev_ind = value - 1
