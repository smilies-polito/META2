from .BaseProblem import BaseProblem
import numpy as np

class BentCigar(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"BentCigar_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return (point[0]**2 + 10**6 * np.sum([point[i]**2 for i in range(1, self.dimensions)]))/10000000