#!/usr/bin/env python
"""Configuring custom git hook path
Creating a separate folder .githooks in the root directory of the project
and add all the hooks there (i.e. pre-commit, prepare-commit-msg, commit-msg, etc.).

Next to update the configurations so that git knows where our hooks:

`git config --global core.hooksPath .githooks`
"""

import os


def main():
    poetry_config = os.path.abspath('pyproject.toml')
    # poetry_config = os.path.abspath("../pyproject.toml")

    with open(poetry_config) as handle:
        lines = handle.readlines()

    i = 0
    for line in lines:
        ind = line.find("=")
        key = line[:ind].strip()
        value = line[ind + 1:].strip()

        if ind > 0 and key == "version":
            version = list(map(int, value.strip('"').split(".")))
            version[2] += 1
            __version__ = ".".join(map(str, version))
            lines[i] = line.replace(value, f'"{__version__}"')
            break

        i += 1

    with open(poetry_config, "w", newline="\n") as handle:
        for line in lines:
            handle.writelines(line)


if __name__ == "__main__":
    main()

# def redo_config(path: str, key: str, value) -> None:
#     with open(path) as handle:
#         lines = handle.readlines()
#
#     i = 0
#     for line in lines:
#         _ind = line.find("=")
#         _key = line[:_ind].strip()
#         _val = line[_ind + 1:].strip()
#
#         if _ind > 0 and _key == key:
#             version = list(map(int, _val.strip('"').split(".")))
#             version[2] += 1
#             __version__ = ".".join(map(str, version))
#             lines[i] = line.replace(_val, f'"{__version__}"')
#             break
#
#         i += 1
#
#     with open(path, "w", newline="\n") as handle:
#         for line in lines:
#             handle.writelines(line)
#
#
# redo_config(poetry_config, "version", )
