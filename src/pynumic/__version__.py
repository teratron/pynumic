from typing import Iterable

VERSION: Iterable[int] = (0, 1, 2)
__version__: str = ".".join(map(str, VERSION))
"""Version."""
