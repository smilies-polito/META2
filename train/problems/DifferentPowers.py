from .BaseProblem import BaseProblem
import numpy as np

class DifferentPowers(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"DifferentPowers_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return np.sqrt(np.sum([abs(point[i])**(2+(4*i/self.dimensions)) for i in range(self.dimensions)]))