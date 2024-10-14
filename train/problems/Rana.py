from .BaseProblem import BaseProblem
import numpy as np

class Rana(BaseProblem):
    def __init__(self, dimensions: int, ranges: list):
        if (len(ranges) == 1):
            ranges = [ranges[0]] * dimensions
        self.name = f"Rana_{dimensions}d"
        self.dimensions = dimensions
        self.ranges = ranges
    
    def get_value(self, point: np.ndarray) -> float:
        result = 0
        for i, x in enumerate(point):
            if i < len(point) - 1:
                result += (x + point[i+1] + (x+1 - np.cos(np.sqrt(abs(point[i+1] + x+1))) * np.sin(np.sqrt(abs(point[i+1] - x+1)))) + 
                        point[i+1] * (point[i+1] + 1 - np.cos(np.sqrt(abs(point[i+1] - x+1))) * np.sin(np.sqrt(abs(point[i+1] + x+1)))))
        return result/20