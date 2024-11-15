import random
from problems.BaseProblem import BaseProblem
import numpy as np

#Takes a problem, randomly rotate and traslate it
def get_linear_transform(range_old, range_new):
    dimensions = len(range_new)
    starting_point = np.array([r[0] for r in range_old][:dimensions])
    range_old = np.array([r[1]-r[0] for r in range_old][:dimensions])
    starting_point_new = np.array([r[0] for r in range_new])
    range_new = np.array([r[1]-r[0] for r in range_new])
    return dimensions, starting_point, range_old, range_new, starting_point_new

def linear_transform(point:np.ndarray, dimensions, starting_point, range_old, range_new, starting_point_new):
    point = point[:dimensions]
    point = point - starting_point 
    point = point / range_old 
    point = point * range_new
    point = point + starting_point_new
    return point 

class Multiplied_problems(BaseProblem):
    def can_rotate(self):
        return self.problem1.can_rotate() and self.problem2.can_rotate()
    
    def __init__(self, problem_1, problem_2):
        self.problem1 = problem_1
        self.problem2 = problem_2
        id_transform = lambda x: x
        self.tranform_problem = None
        self.name = f"{problem_1.get_name()}_mul_{problem_2.get_name()}"
        self.fitness_ranges_1 = problem_1.get_fitness_range_estimate()
        self.fitness_ranges_2 = problem_2.get_fitness_range_estimate()
        assert self.fitness_ranges_1[0] != 0, problem_1.get_name()
        assert self.fitness_ranges_2[0] != 0, problem_2.get_name()
        if (problem_1.get_dimensions() > problem_2.get_dimensions()):
            self.ranges = problem_1.get_ranges()
            self.dimensions = problem_1.get_dimensions()
            self.tranform_problem = 2
            self.dimensions_tr, self.starting_point, self.range_old, self.range_new, self.starting_point_new = get_linear_transform(problem_1.get_ranges(), problem_2.get_ranges())
        else:
            self.ranges = problem_2.get_ranges()
            self.dimensions = problem_2.get_dimensions()
            self.dimensions_tr, self.starting_point, self.range_old, self.range_new, self.starting_point_new = get_linear_transform(problem_2.get_ranges(), problem_1.get_ranges())
            self.tranform_problem = 1

    def get_value(self, point: np.ndarray) -> float: 
        if (self.tranform_problem==1):
            point2 = point 
            point1 = linear_transform(point,self.dimensions_tr,  self.starting_point, self.range_old, self.range_new, self.starting_point_new)
        else:
            point1 = point 
            point2 = linear_transform(point,self.dimensions_tr,  self.starting_point, self.range_old, self.range_new, self.starting_point_new)
        p1_contribution = (self.problem1.get_value(point1) - self.fitness_ranges_1[1])/self.fitness_ranges_1[0]
        p2_contribution = (self.problem2.get_value(point2) - self.fitness_ranges_2[1])/self.fitness_ranges_2[0]
        return p1_contribution * p2_contribution