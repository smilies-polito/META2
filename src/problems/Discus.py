from .BaseProblem import BaseProblem
import numpy as np

class DiscusFunction(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"DiscusFunction_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return (point[0]**2 + 1e6 * np.sum(np.square(point[1:]))) / 1000000