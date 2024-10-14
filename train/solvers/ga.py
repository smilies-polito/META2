import numpy as np
import pygad
from problems.BaseProblem import BaseProblem
from solvers.BaseSolver import BaseSolver

"""Base GA optimizer"""

class GA(BaseSolver):
    def get_name(self) -> str:
        return f"ga-ps_{self.parent_selection_type}-cr_{self.crossover_type}-mp_{self.mutation_p}-mt_{self.mutation_type}"
        
    def __init__(self, problem: BaseProblem, budget=10000,
        parent_selection_type="sss", cr_type="single_point", mutation_p=0.1, pop_size=32, p_mating=6, mutation_type="random",):
        
        self.problem = problem
        self.budget = budget

        #Hyperparameters
        assert parent_selection_type in ["sss", "rws", "rank"]
        assert mutation_p >= 0 and mutation_p <= 1
        assert cr_type in ["single_point","uniform","scattered"]
        assert mutation_type in ["random", "scramble"]
        self.parent_selection_type = parent_selection_type
        self.mutation_p = mutation_p
        self.crossover_type = cr_type
        self.pop_size = pop_size
        self.p_mating = p_mating
        self.mutation_type = mutation_type

        self.num_generations = int(budget/pop_size)
        
    def solve(self):
        init_range_low = [r[0] for r in self.problem.get_ranges()]
        init_range_high = [r[1] for r in self.problem.get_ranges()]
        fitness_func = lambda ga_instance, solution, solution_idx: -self.problem.get_value(np.array(solution))

        gene_space = []
        for i in range(self.problem.get_dimensions()):
            gene_space.append({'low': init_range_low[i], 'high':init_range_high[i]})

        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                        num_parents_mating=self.p_mating,
                        fitness_func=fitness_func,
                        sol_per_pop=self.pop_size,
                        num_genes=self.problem.get_dimensions(),
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=self.parent_selection_type,
                        keep_parents=1,
                        crossover_type=self.crossover_type,
                        mutation_type=self.mutation_type,
                        mutation_probability=self.mutation_p,
                        gene_space=gene_space)
        self.ga_instance.run()
        solution, solution_fitness, _ = self.ga_instance.best_solution()
        return -solution_fitness
    

    def get_variants():
        variants = []
        for mutation in [0.1, 0.2]:
            variants += [
                lambda p, b: GA(p, b, mutation_p=mutation),
                lambda p, b: GA(p, b, mutation_p=mutation, cr_type="uniform"),
                lambda p, b: GA(p, b, mutation_p=mutation, cr_type="scattered"),
                lambda p, b: GA(p, b, mutation_p=mutation, parent_selection_type="rws"),
                lambda p, b: GA(p, b, mutation_p=mutation, parent_selection_type="rank"),
                lambda p, b: GA(p, b, mutation_p=mutation, mutation_type="scramble"),
            ]
        return variants

    def solve_with_starting_population(starting_population, problem, PARENT_SELECTION_TYPE="sss",CR_TYPE="single_point"):
        init_range_low = [r[0] for r in problem.get_ranges()]
        init_range_high = [r[1] for r in problem.get_ranges()]
        gene_space = [{'low': init_range_low[i], 'high':init_range_high[i]} for i in range(problem.get_dimensions())]        
        fitness_func = lambda ga_instance, solution, solution_idx: -problem.get_value(np.array(solution))
        ga_instance = pygad.GA(num_generations=1,
                        num_parents_mating=6,
                        fitness_func=fitness_func,
                        sol_per_pop=len(starting_population),
                        num_genes=problem.get_dimensions(),
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=PARENT_SELECTION_TYPE,
                        keep_parents=1,
                        crossover_type=CR_TYPE,
                        mutation_type="random",
                        mutation_percent_genes=1,
                        gene_space=gene_space)
        ga_instance.population = starting_population
        ga_instance.run()
        return ga_instance.last_generation_fitness
