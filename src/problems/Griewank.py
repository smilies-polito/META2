from .BaseProblem import BaseProblem
import numpy as np

class Griewank(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Griewank_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        sum_term = 0
        prod_term = 1
        for i, x in enumerate(point):
            sum_term += x**2 / 4000
            prod_term *= np.cos(x / np.sqrt(i + 1))
        return sum_term - prod_term + 1