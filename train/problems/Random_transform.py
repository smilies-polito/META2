import random
from problems.BaseProblem import BaseProblem
import numpy as np

#Takes a problem, randomly rotate and traslate it

class Random_transform(BaseProblem):
    def __init__(self, problem, max_traslation_percentage):
        assert problem.get_dimensions() > 1, "The problem must have more than 1 dimension"
        self.problem = problem
        self.rotation = np.random.uniform(2*np.pi)
        self.traslation = np.random.uniform(-max_traslation_percentage, max_traslation_percentage, problem.dimensions)
        self.traslation_real_value = np.array([(r[1]-r[0]) * self.traslation[i] for i,r in enumerate(problem.get_ranges())])
        self.name = f"{problem.get_name()}_Random_transform"
        self.dimensions = problem.get_dimensions()
        self.ranges = list(np.array(problem.get_ranges()) - (np.array([np.array([x,x]) for x in self.traslation_real_value])))

        self.rotation_matrix = np.array(
            [[1]+[0]*(self.problem.dimensions-1)]*(self.problem.dimensions-2) + [
            [0]*(self.problem.dimensions-2) + [np.cos(self.rotation), -np.sin(self.rotation)],
            [0]*(self.problem.dimensions-2) + [np.sin(self.rotation), np.cos(self.rotation)]
        ])

    def can_rotate(self):
        return self.problem.can_rotate()

    def get_value(self, point: np.ndarray) -> float: 
        if (self.problem.can_rotate()):
            rotated_point = np.dot(self.rotation_matrix, point)
        else:
            rotated_point = point
        traslated_point = rotated_point + self.traslation_real_value
        return self.problem.get_value(traslated_point)