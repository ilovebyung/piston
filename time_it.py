import time
from datetime import datetime, date, timedelta


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(f"{func.__name__} ran in {end - start} seconds")
        measured_time = timedelta((end - start)/24/60/60)
        print(f'{measured_time.total_seconds()} seconds')
        return result
    return wrapper


@time_it
def some_function():
    # function logic here
    time.sleep(1.1)


some_function()
