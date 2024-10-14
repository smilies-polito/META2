from .BaseProblem import BaseProblem
import numpy as np

class LinearSlope(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"LinearSlope_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
        self.slopes = np.random.uniform(-1, 1, dimensions)
    
    def get_value(self, point: np.ndarray) -> float:
        return np.sum([self.slopes[i] * point[i] for i in range(self.dimensions)])
    
