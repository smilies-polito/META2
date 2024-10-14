from .BaseProblem import BaseProblem
import numpy as np

class Schwefel(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Schwefel_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return 418.9829 * self.dimensions - np.sum(point * np.sin(np.sqrt(np.abs(point))))