import time
import jax


def pseudo_rn():
    return jax.random.PRNGKey(int(time.perf_counter()))


class Timer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.elapsed_time = -1

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *args, **kwargs):
        self.elapsed_time = time.perf_counter() - self.start_time


timer = Timer()
