import numpy as np
from beecolpy import *
from problems.BaseProblem import BaseProblem
from solvers.BaseSolver import BaseSolver

"""Base ABC optimizer"""
class ABC(BaseSolver):
    def get_name(self) -> str:
        return "abc"
        
    def __init__(self, problem: BaseProblem, budget=10000):
        self.problem = problem
        self.bounds = [(r[0], r[1]) for r in problem.get_ranges()]
        self.budget = budget
        
    def solve(self):
        fitness_function = lambda solution: self.problem.get_value(np.array(solution))
        callback = None
        self.abc = abc(fitness_function,
              self.bounds,
              colony_size=60,
              scouts=0.5,
              iterations=int(self.budget/60),
              min_max='min',
              nan_protection=True,
              log_agents=True)

        self.abc.fit()
        return fitness_function(self.abc.get_solution())
    
    def get_variants():
        return [lambda p, b: ABC(p, b)]