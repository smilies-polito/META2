import numpy as np
from scipy.optimize import differential_evolution
from problems.BaseProblem import BaseProblem
from solvers.BaseSolver import BaseSolver

class DE(BaseSolver):
    def get_name(self) -> str:
        return f"de_{self.strategy}"
        
    def __init__(self, problem: BaseProblem, budget=10000, pop_size=15, strategy='best1bin'):
        self.problem = problem
        self.bounds = [(r[0], r[1]) for r in problem.get_ranges()]
        assert strategy in ['best1bin','rand1bin', 'best1exp', 'currenttobest1bin','randtobest1bin','randtobest1exp','best2exp']
        self.strategy = strategy
        self.budget = budget
        self.pop_size = pop_size
        
    def solve(self):
        unique_dimensions = len(set([f"{r[0]}-{r[1]}" for r in self.problem.get_ranges()]))
        maxiter = int(self.budget/(self.pop_size*unique_dimensions)) 
        fitness_function = lambda solution: self.problem.get_value(np.array(solution))
        result = differential_evolution(fitness_function, self.bounds, strategy=self.strategy, maxiter=maxiter,popsize=self.pop_size)
        n_evaluations = result.nfev
        return result.fun 
    
    def solve_with_starting_population(starting_population, problem, strategy='best1bin'):
        bounds = [(r[0], r[1]) for r in problem.get_ranges()]
        fitness_function = lambda solution: problem.get_value(np.array(solution))
        result = differential_evolution(fitness_function, bounds, strategy=strategy, maxiter=1, init=starting_population)
        return result.population_energies
    
    def get_variants():
        return [
            lambda p, b: DE(p, b),
            lambda p, b: DE(p, b, pop_size=30),
            lambda p, b: DE(p, b, strategy='currenttobest1bin'),
            lambda p, b: DE(p, b, strategy='best2exp'),
            lambda p, b: DE(p, b, strategy='best2exp',pop_size=30),
        ]

"""
    Variants:
        strategy:
            best1bin
            best1exp
            rand1bin
            randtobest1bin
            randtobest1exp
            currenttobest1bin
            best2exp
            
    Hyperparameters
        mutation 
        recombination
"""