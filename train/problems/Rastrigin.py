from .BaseProblem import BaseProblem
import numpy as np

class Rastrigin(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Rastrigin_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        A = 10
        result = A * len(point)
        for arg in point:
            result += arg**2 - A * np.cos(2 * np.pi * arg)
        return result/10