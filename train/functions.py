
"""
    Defines the set of functions to build the training dataset
"""
from problems.BaseProblem import *
from problems.Schwefel import *
from problems.Ackley import *
from problems.Griewank import *
from problems.Rana import *
from problems.Rastrigin import *
from problems.Rosenbrock import *
from problems.Salomon import *
from problems.Benchmark_v1 import *
from problems.Random_transform import *
from problems.Multiplied_problems import *
from problems.Summed_problems import *
from problems.SinProblem import *
from problems.Ellipsoidal import Ellipsoidal
from problems.LinearSlope import LinearSlope
from problems.Discus import DiscusFunction
from problems.BentCigar import BentCigar
from problems.DifferentPowers import DifferentPowers
from problems.Weierstrass import Weierstrass
from problems.Schaffers import Schaffers
from problems.WrapperForPythonBenchmark import get_functions
from problems.StochasticWrappers import *

import random 

def get_base_functions():
    return [
        Schwefel(2, [[-500, 500]]),Schwefel(4, [[-500, 500]]),Schwefel(6, [[-500, 500]]),
        Schwefel(2, [[-1000, 1000]]),Schwefel(4, [[-1000, 1000]]),Schwefel(6, [[-1000, 1000]]),
        Schwefel(2, [[-50, 50]]), Schwefel(2, [[-10, 10]]),
        Ackley(2, [[-32.768, 32.768]]), Ackley(4, [[-52.768, 52.768]]), Ackley(6, [[-72.768, 72.768]]),
        Ackley(2, [[-320.768, 320.768]]), Ackley(4, [[-100.768, 100.768]]), Ackley(6, [[-120.768, 120.768]]),
        Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4))),
        Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4))),
        Griewank(4, [[-32, 32]]), Griewank(5, [[-32, 32]]),
        Griewank(4, [[-64, 64]]), Griewank(5, [[-64, 64]]),
        Rana(2, [[-20, 20]]), Rana(4, [[-20, 20]]), Rana(6, [[-20, 20]]),
        Rana(2, [[-60, 60]]), Rana(4, [[-60, 60]]),
        Rana(4, [[-120, 120]]), Rana(6, [[-120, 120]]),
        Rastrigin(4, [[-30, 30]]), Rastrigin(4, [[-40, 40]]), Rastrigin(6, [[-40, 40]]),Rastrigin(2, [[-60, 60]]),
        Rosenbrock(2, [[-1, 1]]),Rosenbrock(4, [[-1, 1]]), #Only 2 and 4
        Salomon(2, [[-5, 5]]),Salomon(4, [[-5, 5]]),Salomon(6, [[-8, 8]]),Salomon(6, [[-5, 5]]),
        Benchmark_v1(),
        SinProblem(2, [[-10, 10]]), SinProblem(4, [[-10, 10]]), SinProblem(6, [[-10, 10]]),
        SinProblem(4, [[-30, 30]]), SinProblem(6, [[-30, 30]]),
        Schaffers(2, [[-3, 3]]), Schaffers(2, [[-2, 2]]), Schaffers(2, [[-10, 10]]),
        Weierstrass(2, [[-10, 10]]), Weierstrass(2, [[-4, 4]]),Weierstrass(2, [[-1, 1]]),
        Weierstrass(4, [[-10, 10]]), Weierstrass(4, [[-4, 4]]),Weierstrass(4, [[-1, 1]]),
        Weierstrass(6, [[-10, 10]]), Weierstrass(6, [[-4, 4]]),Weierstrass(6, [[-1, 1]]),
        DifferentPowers(2, [[-10, 10]]), DifferentPowers(6, [[-10, 10]]),
        DifferentPowers(2, [[-20, 20]]), DifferentPowers(4, [[-20, 20]]),
        BentCigar(2, [[-5, 5]]), BentCigar(4, [[-5, 5]]),BentCigar(6, [[-5, 5]]), 
        DiscusFunction(2, [[-5, 5]]), DiscusFunction(4, [[-5, 5]]), DiscusFunction(6, [[-5, 5]]),
        LinearSlope(2, [[-50, 50]]), LinearSlope(4, [[-50, 50]]), LinearSlope(6, [[-50, 50]]),
        Ellipsoidal(2, [[-30, 30]]), Ellipsoidal(4, [[-30, 30]]), Ellipsoidal(6, [[-30, 30]])
    ] + get_functions()

def get_split_functions():
    training_functions = [
        Schwefel(2, [[-500, 500]]),Schwefel(4, [[-500, 500]]),Schwefel(6, [[-500, 500]]),
        Schwefel(2, [[-1000, 1000]]),Schwefel(4, [[-1000, 1000]]),Schwefel(6, [[-1000, 1000]]),
        Schwefel(2, [[-50, 50]]), Schwefel(2, [[-10, 10]]),
        Ackley(2, [[-32.768, 32.768]]), Ackley(4, [[-52.768, 52.768]]), Ackley(6, [[-72.768, 72.768]]),
        Ackley(2, [[-320.768, 320.768]]), Ackley(4, [[-100.768, 100.768]]), Ackley(6, [[-120.768, 120.768]]),
        Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4))),
        Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Summed_problems(Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4), Random_transform(Ackley(4, [[-92.768, 92.768]]),0.4))),
        Griewank(4, [[-32, 32]]), Griewank(5, [[-32, 32]]),
        Griewank(4, [[-64, 64]]), Griewank(5, [[-64, 64]]),
        Rana(2, [[-20, 20]]), Rana(4, [[-20, 20]]), Rana(6, [[-20, 20]]),
        Rana(2, [[-60, 60]]), Rana(4, [[-60, 60]]),
        Rana(4, [[-120, 120]]), Rana(6, [[-120, 120]]),
        Rastrigin(4, [[-30, 30]]), Rastrigin(4, [[-40, 40]]), Rastrigin(6, [[-40, 40]]),Rastrigin(2, [[-60, 60]]),
        Rosenbrock(2, [[-1, 1]]),Rosenbrock(4, [[-1, 1]]), #Only 2 and 4
        Salomon(2, [[-5, 5]]),Salomon(4, [[-5, 5]]),Salomon(6, [[-8, 8]]),Salomon(6, [[-5, 5]]),
        Benchmark_v1()
    ]
    test_functions = [
        SinProblem(2, [[-10, 10]]), SinProblem(4, [[-10, 10]]), SinProblem(6, [[-10, 10]]),
        SinProblem(4, [[-30, 30]]), SinProblem(6, [[-30, 30]]),
        Schaffers(2, [[-3, 3]]), Schaffers(2, [[-2, 2]]), Schaffers(2, [[-10, 10]]),
        Weierstrass(2, [[-10, 10]]), Weierstrass(2, [[-4, 4]]),Weierstrass(2, [[-1, 1]]),
        Weierstrass(4, [[-10, 10]]), Weierstrass(4, [[-4, 4]]),Weierstrass(4, [[-1, 1]]),
        Weierstrass(6, [[-10, 10]]), Weierstrass(6, [[-4, 4]]),Weierstrass(6, [[-1, 1]]),
        DifferentPowers(2, [[-10, 10]]), DifferentPowers(6, [[-10, 10]]),
        DifferentPowers(2, [[-20, 20]]), DifferentPowers(4, [[-20, 20]]),
        BentCigar(2, [[-5, 5]]), BentCigar(4, [[-5, 5]]),BentCigar(6, [[-5, 5]]), 
        DiscusFunction(2, [[-5, 5]]), DiscusFunction(4, [[-5, 5]]), DiscusFunction(6, [[-5, 5]]),
        LinearSlope(2, [[-50, 50]]), LinearSlope(4, [[-50, 50]]), LinearSlope(6, [[-50, 50]]),
        Ellipsoidal(2, [[-30, 30]]), Ellipsoidal(4, [[-30, 30]]), Ellipsoidal(6, [[-30, 30]])
    ]
    #pybenchfunction is a set of 130 functions. Some are variants of functions already in the training set
    #To ensure a clean validation set, these functions are forced to be in the training dataset as well, 
    #while the others are split between the two sets
    pybenchmark_functions = get_functions()
    free_functions = []
    training_set_names = set(['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Salomon', 'Schwefel', 'Schwefel 2.20', 'Schwefel 2.21', 'Schwefel 2.22', 'Schwefel 2.23', 'Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Salomon', 'Schwefel', 'Schwefel 2.20', 'Schwefel 2.21', 'Schwefel 2.22', 'Schwefel 2.23', 'Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Salomon', 'Schwefel', 'Schwefel 2.20', 'Schwefel 2.21', 'Schwefel 2.22', 'Schwefel 2.23'])
    for function in pybenchmark_functions:
        if function.benchmark.name in training_set_names:
            training_functions.append(function)
        else:
            free_functions.append(function)
    training_functions += free_functions[:80]
    test_functions += free_functions[80:]
    return training_functions, test_functions

#Perform data augmentation
def augment_functions(functions):
    functions = [Random_transform(problem, 0.3) for problem in functions*2]
    N = len(functions)*4
    for i in range(N):
        f1, f2 = random.choice(functions), random.choice(functions)
        functions.append(Summed_problems(f1, f2, weight_1=random.random()*0.7+0.15))
        f1, f2 = random.choice(functions), random.choice(functions)
        functions.append(Multiplied_problems(f1, f2))
    return functions

