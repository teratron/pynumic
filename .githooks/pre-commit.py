#!/usr/bin/env python
"""Configuring custom git hook path
Creating a separate folder .githooks in the root directory of the project
and add all the hooks there (i.e. pre-commit, prepare-commit-msg, commit-msg, etc.).

Next to update the configurations so that git knows where our hooks:

`git config --global core.hooksPath .githooks`
"""

import os
from typing import Callable, Any


# def main():
#     poetry_config = os.path.abspath('pyproject.toml')
#     # poetry_config = os.path.abspath("../pyproject.toml")
#
#     with open(poetry_config) as handle:
#         lines = handle.readlines()
#
#     i = 0
#     for line in lines:
#         ind = line.find("=")
#         key = line[:ind].strip()
#         value = line[ind + 1:].strip()
#
#         if ind > 0 and key == "version":
#             version = list(map(int, value.strip('"').split(".")))
#             version[2] += 1
#             __version__ = ".".join(map(str, version))
#             lines[i] = line.replace(value, f'"{__version__}"')
#             break
#
#         i += 1
#
#     with open(poetry_config, "w", newline="\n") as handle:
#         for line in lines:
#             handle.writelines(line)
#
#
# if __name__ == "__main__":
#     main()


def get_conf_value(path: str, key: str) -> (str | None, list[str]):
    with open(path) as handle:
        lines = handle.readlines()

    for line in lines:
        _ind = line.find("=")
        _key = line[:_ind].strip()
        _val = line[_ind + 1:].strip()

        if _ind > 0 and _key == key:
            return _val, lines

    return None,


def set_conf_value(path: str, key: str, value: str | Callable[[str, dict[str, Any]], str], **options: Any) -> None:
    # _val, lines = get_conf_value(path, key)
    # if _val is not None:
    #     __val = ""
    #     if isinstance(value, str):
    #         __val = value
    #     elif isinstance(value, Callable):
    #         __val = value(_val, **options)
    #     else:
    #         raise TypeError("error")
    #
    #     lines[i] = lines[i].replace(_val, __val)

    with open(path) as handle:
        lines = handle.readlines()

    i = 0
    for line in lines:
        _ind = line.find("=")
        _key = line[:_ind].strip()
        _val = line[_ind + 1:].strip()

        if _ind > 0 and _key == key:
            __val = ""
            if isinstance(value, str):
                __val = value
            elif isinstance(value, Callable):
                __val = value(_val, **options)
            else:
                raise TypeError("error")

            lines[i] = line.replace(_val, __val)
            break
        i += 1

    with open(path, "w", newline="\n") as handle:
        for line in lines:
            handle.writelines(line)


def increase_version(value: str, *, w: int = 2) -> str:
    if w < 0 or w > 2:
        raise IndexError("error")

    version = list(map(int, value.strip('"').split(".")))
    version[w] += 1

    match w:
        case 1:
            version[2] = 0
        case 0:
            version[2] = version[1] = 0

    return f'"{".".join(map(str, version))}"'


set_conf_value(os.path.abspath("../pyproject.toml"), "version", increase_version)
# set_conf_value(os.path.abspath("../pyproject.toml"), "version", increase_version, w=0)
