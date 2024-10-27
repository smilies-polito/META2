from solvers.ga import GA
from solvers.de import DE
from solvers.abc import ABC
from solvers.pso import PSO

def get_algorithms_lambdas():
    return GA.get_variants()+DE.get_variants()+ABC.get_variants()+PSO.get_variants()

def get_algorithms_names():
    return [f"GA_{i+1}" for i in range(len(GA.get_variants()))] + [f"DE_{i+1}" for i in range(len(DE.get_variants()))] + [f"ABC_{i+1}" for i in range(len(ABC.get_variants()))] + [f"PSO_{i+1}" for i in range(len(PSO.get_variants()))]