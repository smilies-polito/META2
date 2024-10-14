import numpy as np
import pyswarms as ps
from problems.BaseProblem import BaseProblem
from solvers.BaseSolver import BaseSolver
import sys, os

class PSO(BaseSolver):
    def get_name(self) -> str:
        return f"pso-particles_{self.n_particles}-c1_{self.cognitive_coefficient}-c2_{self.social_coefficient}-w_{self.intertia}"
        
    def __init__(self, problem: BaseProblem, budget=10000, 
        n_particles=10, cognitive_coefficient=0.5, social_coefficient=0.3, intertia=0.9,
        initial_solution=None):
        self.problem = problem
        self.bounds = (np.array([r[0] for r in problem.get_ranges()]), np.array([r[1] for r in problem.get_ranges()]))
        self.initial_solution = initial_solution
        # Set-up hyperparameters
        self.n_iter = int(budget/n_particles)
        self.n_particles = n_particles
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.intertia = intertia


    def solve(self):
        options = {'c1':self.cognitive_coefficient, 'c2':self.social_coefficient, 'w':self.intertia}

        self.model = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.problem.get_dimensions(), options=options, bounds=self.bounds)
        if self.initial_solution is not None:
            self.model.pos = np.array(self.initial_solution)

        fitness_function = lambda solutions: [self.problem.get_value(np.array(solution)) for solution in solutions]
        cost, pos = self.model.optimize(fitness_function, iters=self.n_iter,verbose=False)
        return cost


    def local_vs_globalperformance(problem: BaseProblem):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        bounds = (np.array([r[0] for r in problem.get_ranges()]), np.array([r[1] for r in problem.get_ranges()]))
        model_g = ps.single.GlobalBestPSO(n_particles=10, dimensions=problem.get_dimensions(), options=options, bounds=bounds)
        fitness_function = lambda solutions: [problem.get_value(np.array(solution)) for solution in solutions]
        cost_global, _ = model_g.optimize(fitness_function, iters=1,verbose=False)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        model_l = ps.single.LocalBestPSO(n_particles=10, dimensions=problem.get_dimensions(), options=options, bounds=bounds)
        cost_local, _ = model_l.optimize(fitness_function, iters=1,verbose=False)
        return cost_global - cost_local
        
    def get_variants():
        variants = []
        for n_particles in [10, 30]:
            variants += [
            lambda p, b: PSO(p, b, n_particles=n_particles),
            lambda p, b: PSO(p, b, social_coefficient=0.5, n_particles=n_particles),
            lambda p, b: PSO(p, b, social_coefficient=0.1, n_particles=n_particles),
            ]
        return variants