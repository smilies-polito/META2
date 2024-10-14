from .BaseProblem import BaseProblem
import numpy as np

class Rosenbrock(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Rosenbrock_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        sum_term = 0
        for i in range(len(point) - 1):
            sum_term += 100 * (point[i+1] - point[i]**2)**2 + (point[i] - 1)**2
        return sum_term