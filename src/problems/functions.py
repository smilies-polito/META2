
"""
    Defines the set of functions to build the training dataset
"""
from .BaseProblem import *
from .Schwefel import *
from .Ackley import *
from .Griewank import *
from .Rana import *
from .Rastrigin import *
from .Rosenbrock import *
from .Salomon import *
from .Benchmark_v1 import *
from .Random_transform import *
from .Multiplied_problems import *
from .Summed_problems import *
from .SinProblem import *
from .Ellipsoidal import Ellipsoidal
from .LinearSlope import LinearSlope
from .Discus import DiscusFunction
from .BentCigar import BentCigar
from .DifferentPowers import DifferentPowers
from .Weierstrass import Weierstrass
from .Schaffers import Schaffers
from .WrapperForPythonBenchmark import get_functions
from .StochasticWrappers import *
import random 
import re 

DEFINED_FUNCTIONS = [
        Schwefel(2, [[-500, 500]]),Schwefel(4, [[-500, 500]]),Schwefel(6, [[-500, 500]]),
        Schwefel(2, [[-1000, 1000]]),Schwefel(4, [[-1000, 1000]]),Schwefel(6, [[-1000, 1000]]),
        Schwefel(2, [[-50, 50]]), Schwefel(2, [[-10, 10]]),
        Ackley(2, [[-32.768, 32.768]]), Ackley(4, [[-52.768, 52.768]]), Ackley(6, [[-72.768, 72.768]]),
        Ackley(2, [[-320.768, 320.768]]), Ackley(4, [[-100.768, 100.768]]), Ackley(6, [[-120.768, 120.768]]),
        Griewank(4, [[-32, 32]]), Griewank(5, [[-32, 32]]),
        Griewank(4, [[-64, 64]]), Griewank(5, [[-64, 64]]),
        Griewank(4, [[-10, 10]]), Griewank(5, [[-10, 10]]),
        Griewank(4, [[-100, 100]]), Griewank(5, [[-100, 100]]),
        Rana(2, [[-20, 20]]), Rana(4, [[-20, 20]]), Rana(6, [[-20, 20]]),
        Rana(2, [[-60, 60]]), Rana(4, [[-60, 60]]),
        Rana(4, [[-120, 120]]), Rana(6, [[-120, 120]]),
        Rastrigin(4, [[-30, 30]]), Rastrigin(4, [[-40, 40]]), Rastrigin(6, [[-40, 40]]),Rastrigin(2, [[-60, 60]]),
        Rastrigin(4, [[-5, 5]]), Rastrigin(4, [[-5, 5]]),
        Rosenbrock(2, [[-1, 1]]),Rosenbrock(4, [[-1, 1]]), #Only 2 and 4
        Salomon(2, [[-5, 5]]),Salomon(4, [[-5, 5]]),Salomon(6, [[-8, 8]]),Salomon(6, [[-5, 5]]),
        Benchmark_v1(),
        SinProblem(2, [[-5, 5]]), SinProblem(4, [[-5, 5]]), SinProblem(6, [[-5, 5]]),
        SinProblem(2, [[-10, 10]]), SinProblem(4, [[-10, 10]]), SinProblem(6, [[-10, 10]]),
        SinProblem(4, [[-30, 30]]), SinProblem(6, [[-30, 30]]),
        Schaffers(2, [[-3, 3]]), Schaffers(2, [[-2, 2]]), Schaffers(2, [[-10, 10]]),
        Schaffers(2, [[-1, 1]]), Schaffers(2, [[0, 10]]), Schaffers(2, [[-1, 3]]),
        Weierstrass(2, [[-10, 10]]), Weierstrass(2, [[-4, 4]]),Weierstrass(2, [[-1, 1]]),
        Weierstrass(4, [[-10, 10]]), Weierstrass(4, [[-4, 4]]),Weierstrass(4, [[-1, 1]]),
        Weierstrass(6, [[-10, 10]]), Weierstrass(6, [[-4, 4]]),Weierstrass(6, [[-1, 1]]),
        DifferentPowers(2, [[-10, 10]]), DifferentPowers(6, [[-10, 10]]),
        DifferentPowers(2, [[-20, 20]]), DifferentPowers(4, [[-20, 20]]),
        BentCigar(2, [[-5, 5]]), BentCigar(4, [[-5, 5]]),BentCigar(6, [[-5, 5]]), 
        DiscusFunction(2, [[-5, 5]]), DiscusFunction(4, [[-5, 5]]), DiscusFunction(6, [[-5, 5]]),
        LinearSlope(2, [[-50, 50]]), LinearSlope(4, [[-50, 50]]), LinearSlope(6, [[-50, 50]]),
        LinearSlope(2, [[-10, 10]]), LinearSlope(4, [[-10, 10]]), LinearSlope(6, [[-10, 10]]),
        Ellipsoidal(2, [[-30, 30]]), Ellipsoidal(4, [[-30, 30]]), Ellipsoidal(6, [[-30, 30]]),
        Ellipsoidal(2, [[-10, 10]]), Ellipsoidal(4, [[-10, 10]]), Ellipsoidal(6, [[-10, 10]]),
        Ellipsoidal(2, [[-70, 70]]), Ellipsoidal(4, [[-70, 70]]), Ellipsoidal(6, [[-70, 70]])
    ]

def get_base_functions():
    return DEFINED_FUNCTIONS + get_functions()

def get_function_test(seed):
    def clean_string(input_string):
        cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
        return cleaned_string.lower()
    def is_included(string1, string2):
        return string1 in string2 or string2 in string1

    def split_functions_in_disjointed_sets():
        functions = DEFINED_FUNCTIONS
        functions = [(f, clean_string(f.__class__.__name__)) for f in functions]
        functions += [(f, clean_string(f.benchmark.name)) for f in get_functions()]
        sets = []
        for i in range(len(functions)):
            f = functions[i]
            incl = False
            for f2 in sets:
                if is_included(f[1], f2[0]):
                    incl = True
                    f2[1].append(f)
                    if len(f[1])>len(f2[0]):
                        f2[0] = f[1]
                    break
            if incl == False:
                sets.append(
                    [f[1], [f]]
                )
        return sets 
            
            
    f = split_functions_in_disjointed_sets()
    s1, s2 = [], []
    random.seed(seed)
    random.shuffle(f)
    for c in f:
        if len(s2)>(len(s1)+len(s2))*0.17:
            s1 += c[1]
        else:
            s2 += c[1]
    n1, n2 = [f[1] for f in s1], [f[1] for f in s2]
    s1 = [problem[0] for problem in s1]
    s2 = [problem[0] for problem in s2]
    return s1, n1, s2, n2

#Perform data augmentation
def augment_functions(base_functions):
    functions =  [Random_transform(problem, 0.3) for problem in base_functions*2]
    N = len(base_functions)*5
    for i in range(N):
        f1, f2 = random.choice(functions), random.choice(functions)
        functions.append(Summed_problems(f1, f2, weight_1=random.random()*0.7+0.15))
        f1, f2 = random.choice(functions), random.choice(functions)
        functions.append(Multiplied_problems(f1, f2))
    
    function_set_2 = [Random_transform(problem, 0.3) for problem in base_functions]
    N = len(base_functions)*6
    for i in range(N):
        f1, f2 = random.choice(function_set_2), random.choice(base_functions)
        function_set_2.append(Summed_problems(f1, f2, weight_1=random.random()*0.7+0.15))
        f1, f2 = random.choice(function_set_2), random.choice(base_functions)
        function_set_2.append(Multiplied_problems(f1, f2))
    function_set_3 = [Random_transform(problem, 0.3) for problem in base_functions]
    N = len(base_functions)*6
    for i in range(N):
        f1, f2 = random.choice(function_set_3), random.choice(base_functions)
        function_set_3.append(Summed_problems(f1, f2, weight_1=random.random()*0.7+0.15))
        f1, f2 = random.choice(function_set_3), random.choice(base_functions)
        function_set_3.append(Multiplied_problems(f1, f2))
    return functions+function_set_2+function_set_3

