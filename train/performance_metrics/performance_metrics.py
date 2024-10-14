import numpy as np


#Raw scores, array in form [[fitness for each F] for each A]
def full_comparison_oriented_scores(raw_scores):
    for function_i in len(raw_scores[0]):
        solutions = [raw_scores[i][function_i] for i in range(len(raw_scores))]
        #Get best and worst solutions
        best_solution = min(solutions)
        worst_solution = max(solutions)
        #Define linear transform
        if (best_solution!=worst_solution):
            transform = lambda x: (x - worst_solution) / (worst_solution - best_solution) * -1
        else:
            transform = lambda _: 1
        #Compute metrics
        for i in range(len(raw_scores)):
            raw_scores[i][function_i] = transform(raw_scores[i][function_i])
    return raw_scores

