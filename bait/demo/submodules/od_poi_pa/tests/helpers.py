from functools import wraps
from time import time


def measure_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__!r} executed in {(end-start):.3f} seconds")
        return result

    return wrapper
