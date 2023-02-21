from typing import Any


def set_props(obj: object, *args: Any, **kwargs: Any) -> None:
    print(obj, args, kwargs)
