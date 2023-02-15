from typing import Iterable


VERSION: Iterable[int] = (0, 1, 1)
__version__: str = ".".join(map(str, VERSION))
"""Version."""
