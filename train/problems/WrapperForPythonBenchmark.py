from .BaseProblem import BaseProblem
import sys
sys.path.insert(0, 'problems/')
import pybenchfunction as bench
import numpy as np

#Wrapper for the functions provided by the PythonBenchmark repository
class WrapperForPythonBenchmark(BaseProblem):
    def __init__(self, benchmark, dimensions):
        self.benchmark = benchmark
        self.name = "wr_" + benchmark.name
        self.dimensions = dimensions
        self.ranges = benchmark.input_domain
        self.scale = np.max(np.array([1]+[np.abs(self.benchmark(s)) for s in self.sample_uniform(500)]))
        assert benchmark.is_dim_compatible(dimensions)
    
    def get_value(self, X):
        for k in range(len(X)):
            X[k] = max(min(X[k], self.ranges[k][1]), self.ranges[k][0])
        return self.benchmark(X)/self.scale
        
    def can_rotate(self):
        return False
    

def get_functions():
    functions = []
    for i in range(2,5):
        f_i = bench.get_functions(i, continuous=True)
        f_i = [WrapperForPythonBenchmark(f(i), i) for f in f_i]
        functions.extend(f_i)
    return functions
