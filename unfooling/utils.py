import time
import math
from contextlib import contextmanager
from datetime import timedelta

import numpy as np


@contextmanager
def timer(name):
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        print(f'{name} took {str(timedelta(seconds=dur))}')


if hasattr(math, 'prod'):  # available in 3.8+
    prod = math.prod
else:  # functionally equivalent w/o positional argument checking
    """
    >>> %timeit reduce(mul, values)
    180 µs ± 2.15 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit math.prod(values)
    133 µs ± 1.57 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> math.prod(values) == reduce(mul, values)
    True
    """
    import operator
    from functools import reduce


    def prod(iterable, start=1):
        return reduce(operator.mul, iterable, start)


def debug_print(x, name):
    print(f'{name} min={x.min()} max={x.max()} '
          f'pct0={(x == 0).sum() / x.size * 100:.2f}')


def inf_(op, x, *args, **kwargs):
    with np.errstate(divide='ignore', over='ignore'):
        x = op(x, *args, **kwargs)
        mask = np.isfinite(x)
        x_infinite = x[~mask]
        signs = np.sign(x_infinite)
        neg = signs < 0
        x_infinite[neg] = np.finfo(x.dtype).min
        x_infinite[~neg] = np.finfo(x.dtype).max
        x[~mask] = x_infinite
    return x
