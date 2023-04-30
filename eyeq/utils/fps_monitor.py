import time
from threading import Lock


class FpsMonitor:

    def __init__(self):
        self.processed_times = list()
        self.last_timestamp = None
        self.lock = Lock()

    def tick(self):
        now = time.time()
        if self.last_timestamp:
            elapsed_time = now - self.last_timestamp
            if len(self.processed_times) > 30:
                self.processed_times.pop(0)
            self.processed_times.append(elapsed_time)
        self.last_timestamp = now

    def get_fps(self):
        """
        :return: average fps of last 30 ticks
        """
        with self.lock:
            processed_times = self.processed_times
            return len(processed_times) / sum(processed_times)


