from .BaseProblem import BaseProblem
import numpy as np

class Weierstrass(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Weierstrass_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        return np.sum([np.sum([0.5**i * np.cos(2*np.pi*10**i*(point[j]+0.5)) for i in range(20)]) for j in range(self.dimensions)]) - self.dimensions * np.sum([0.5**i * np.cos(2*np.pi*10**i*0.5) for i in range(20)])