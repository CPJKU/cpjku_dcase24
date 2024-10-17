from time import perf_counter
import datasets

logger = datasets.logging.get_logger(__name__)


class catchtime:
    # context to measure loading time: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    def __init__(self, debug_print="Time", logger=logger):
        self.debug_print = debug_print
        self.logger = logger

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        readout = f"{self.debug_print}: {self.time:.3f} seconds"
        self.logger.info(readout)
