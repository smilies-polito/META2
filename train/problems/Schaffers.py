from .BaseProblem import BaseProblem
import numpy as np

class Schaffers(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Schaffers_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, X: np.ndarray) -> float:
        X = np.sum(X**2)
        num = (np.sin((X**2)**2)**2) - 0.5
        den = (1 + 0.001*(X**2))**2 
        return 0.5 + num/den