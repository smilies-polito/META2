from .BaseProblem import BaseProblem
from scipy.stats import truncnorm
import numpy as np

class StochasticWrapperMultiplier(BaseProblem):
    def __init__(self, problem):
        self.name = f"ND_{problem.get_name()}"
        self.dimensions = problem.get_dimensions()
        self.ranges = problem.get_ranges()
        self.problem = problem
        self.N = 1+(np.random.random()/2)
    
    def get_value(self, point: np.ndarray) -> float:
        return self.problem.get_value(point) * (2-self.N) + (np.random.random() * (self.N * 2 - 2) )  
    
    def can_rotate(self):
        return self.problem.can_rotate()

class StochasticWrapperNormal(BaseProblem):
    def __init__(self, problem):
        self.name = f"ND_{problem.get_name()}"
        self.dimensions = problem.get_dimensions()
        self.ranges = problem.get_ranges()
        self.problem = problem
        #Find MIN estimate
        samples = problem.sample_uniform(500)
        solutions = [problem.get_value(i) for i in samples]
        self.minimum, maximum = min(solutions), max(solutions)
        self.N = (maximum - self.minimum)*0.05
        
    
    def get_value(self, point: np.ndarray) -> float:
        value = max(0,(self.problem.get_value(point)-self.minimum))
        a, b = (-value) / self.N, np.inf
        trunc_dist = truncnorm(a, b, loc=value, scale=self.N)
        return trunc_dist.rvs() 

    def can_rotate(self):
        return self.problem.can_rotate()

class StochasticWrapperSummer(BaseProblem):
    def __init__(self, problem):
        self.name = f"ND_{problem.get_name()}"
        self.dimensions = problem.get_dimensions()
        self.ranges = problem.get_ranges()
        self.problem = problem
        #Get some idea of function range
        samples = problem.sample_uniform(100)
        values = [problem.get_value(i) for i in samples]
        self.interval = np.percentile(values, 90) - np.percentile(values, 10)
        self.interval *= np.random.random()*0.3
    
    def get_value(self, point: np.ndarray) -> float:
        return self.problem.get_value(point) + (self.interval * (1-(np.random.random()*2)))
    
    def can_rotate(self):
        return self.problem.can_rotate()