from datetime import datetime
from functools import wraps
from time import time


def human_readable_time(seconds: int):
    if seconds < 0:
        raise ValueError("Seconds cannot be negative")

    if seconds >= 86400:  # 86400 seconds in a day
        return f"{seconds // 86400:.0f} day{'s' if seconds // 86400 > 1 else ''}"
    elif seconds >= 3600:  # 3600 seconds in an hour
        return f"{seconds // 3600:.0f} hour{'s' if seconds // 3600 > 1 else ''}"
    elif seconds >= 60:  # 60 seconds in a minute
        return f"{seconds // 60:.0f} minute{'s' if seconds // 60 > 1 else ''}"
    else:
        return f"{seconds:.0f} second{'s' if seconds != 1 else ''}"


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time} seconds")
        return result

    return timeit_wrapper


def get_current_date():
    return datetime.now().strftime("%d/%m/%Y")
