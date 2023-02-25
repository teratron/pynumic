import os
from dataclasses import dataclass

ROOT: str = os.path.join("/")
# ROOT: str = sys.argv[0].rstrip("/setup.py")
SRC: str = os.path.join(ROOT, "..")
TESTS: str = os.path.join(ROOT, "../../tests")


# print(__file__)
# print(sys.argv[0].rstrip("/setup.py"))


@dataclass
class Path:
    # directories
    root: str = os.path.abspath("../..")
    src: str = os.path.join(root, "..")
    tests: str = os.path.join(root, "../../tests")

    # files
    poetry_conf: str = os.path.join(root, "../../pyproject.toml")

# print(Path.root)
# print(Path.src)
