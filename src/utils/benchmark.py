"""Benchmark"""  # TODO:

from time import time
from typing import Any, Callable, TypeAlias

_TargetType: TypeAlias = Callable[[Any, Any], Any]
_InnerType: TypeAlias = Callable[[Any, Any], float]
_OuterType: TypeAlias = Callable[[_TargetType], _InnerType]


def benchmark(repeat: int = 1000, epoch: int = 100) -> _OuterType:
    """Benchmark.
    :param repeat:
    :param epoch:
    :return:
    """

    def outer(func: _TargetType) -> _InnerType:
        def inner(*args: Any, **kwargs: Any) -> float:
            sum_time: float = 0
            for _ in range(epoch):
                start: float = time()

                for _ in range(repeat):
                    # _ = func(*args, **kwargs)
                    func(*args, **kwargs)

                sum_time += time() - start

            bench_time: float = sum_time / epoch
            print(f"benchmark function '{func.__name__}': average time: {bench_time} sec")

            return bench_time

        return inner

    return outer

# @benchmark(epoch=1000)
# def _get_lwma1(data: list[float], period: int) -> list[float]:
#     array: list[float] = []
#
#     for i in range(len(data) - period + 1):
#         n = 0
#         m = 0.0
#         for j in range(i, i + period):
#             k = j + 1
#             n += k
#             m += data[j] * k
#
#         array.append(m / n)
#
#     return array
#
#
# @benchmark(epoch=1000)
# def _get_lwma2(data: list[float], period: int) -> list[float]:
#     length: int = len(data) - period + 1
#     array: list[float] = [0.0] * length
#
#     for i in range(length):
#         n = 0
#         for y in range(i, i + period):
#             z = y + 1
#             n += z
#             array[i] += data[y] * z
#
#         array[i] /= n
#
#     return array
#
#
# if __name__ == "__main__":
#     print(_get_lwma1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3))
#     print(_get_lwma2([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3))
