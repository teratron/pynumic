from typing import Iterable

VERSION: Iterable[int] = (0, 1, 6)
__version__: str = ".".join(map(str, VERSION))
"""Version."""
