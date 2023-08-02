"""TODO: _exceptions.py - Exceptions."""


class PynumicException(Exception):
    """PynumicException."""


class InvalidProjectFile(PynumicException):
    """InvalidProjectFile."""


class PynumicValueError(ValueError):
    """Pynumic value error."""

    _message: str = "---value error---"
    __message: str = ""

    def __init_subclass__(cls) -> None:
        # print(cls._message)
        cls.__message = cls._message

    def __init__(self, message: str = _message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.message in self.__message:
            print("+++")
            return f"{PynumicValueError._message}"

        print("---")
        return f"{self.message} {self._message}"


class NotANumberError(PynumicValueError):
    """Not-a-Number value."""

    _message: str = "not-a-number value"

    # def __init__(self, message: str = _message) -> None:
    #     self.message = message
    #     super().__init__(self.message)


class InfinityError(PynumicValueError):
    """Infinity value."""

    _message: str = "infinity value"

    # def __init__(self, message: str = _message) -> None:
    #     self.message = message
    #     super().__init__(self.message)


if __name__ == "__main__":
    try:
        raise NotANumberError
    except NotANumberError as err:
        print(f"Not a number error: {err}")

    try:
        raise NotANumberError("loss is")
    except NotANumberError as err:
        print(f"{err}")

    try:
        raise InfinityError
    except InfinityError as err:
        print(f"Infinity error: {err}")

    try:
        raise InfinityError("loss is")
    except InfinityError as err:
        print(f"{err}")
