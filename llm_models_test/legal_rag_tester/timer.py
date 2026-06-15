"""Timing utility for measuring sub-step durations."""
import time

class StepTimer:
    """Context manager that measures elapsed ms for any named step."""
    
    def __init__(self, name: str):
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self): 
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000