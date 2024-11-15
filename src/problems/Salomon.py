from .BaseProblem import BaseProblem
import numpy as np

class Salomon(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Salomon_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        squared_sum = np.sum(np.array(point) ** 2)
        sqrt_sum = np.sqrt(squared_sum)
        result = 1 - np.cos(2 * np.pi * sqrt_sum) + 0.1 * sqrt_sum
        return result