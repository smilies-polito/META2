from .BaseProblem import BaseProblem
import numpy as np

class Ellipsoidal(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Ellipsoidal_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return np.sum([((i+1) * point[i])**2 for i in range(self.dimensions)]) / 100000
