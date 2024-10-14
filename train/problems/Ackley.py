from .BaseProblem import BaseProblem
import numpy as np

class Ackley(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Ackley_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(point**2) / len(point)))
        term2 = -np.exp(np.sum(np.cos(c*point)) / len(point))
        return term1 + term2 + a + np.exp(1)