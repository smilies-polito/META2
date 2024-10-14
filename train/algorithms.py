from solvers.ga import GA
from solvers.de import DE
from solvers.abc import ABC
from solvers.pso import PSO

def get_algorithms_lambdas():
    return GA.get_variants()+DE.get_variants()+ABC.get_variants()+PSO.get_variants()