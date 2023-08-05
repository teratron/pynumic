"""TODO: _exceptions.py - Exceptions."""


class PynumicException(Exception):
    """PynumicException."""


class InvalidProjectFile(PynumicException):
    """InvalidProjectFile."""


class PynumicValueError(ValueError):
    """Pynumic value error."""

    _message: str = "value error"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.message in self._message:
            return f"{self._message}"

        return f"{self.message} {self._message}"


class NotANumberError(PynumicValueError):
    """Not a number value."""

    _message: str = "not-a-number value"


class InfinityError(PynumicValueError):
    """Infinity value."""

    _message: str = "infinity value"


if __name__ == "__main__":
    try:
        raise PynumicValueError
    except PynumicValueError as err:
        print(f"{err}")

    try:
        raise PynumicValueError("loss is")
    except PynumicValueError as err:
        print(f"{err}\n")

    try:
        raise NotANumberError
    except NotANumberError as err:
        print(f"{err}")

    try:
        raise NotANumberError("miss is")
    except NotANumberError as err:
        print(f"{err}\n")

    try:
        raise InfinityError
    except InfinityError as err:
        print(f"{err}")

    try:
        raise InfinityError("variable is")
    except InfinityError as err:
        print(f"{err}")
