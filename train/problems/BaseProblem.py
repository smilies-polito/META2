import numpy as np

# Base class for all problems
class BaseProblem:
    def get_name(self) -> str:
        return self.name 

    # Return dimensionality of input 
    def get_dimensions(self) -> int:
        return self.dimensions 

    # Return the ranges of the input
    def get_ranges(self) -> list:
        return self.ranges
    
    # Get the value of the function at a given point
    def get_value(self, point: np.ndarray) -> float:
        pass

    def can_rotate(self):
        return True

    def sample_uniform(self, n_samples):
        samples = []
        for r in self.get_ranges():
            samples.append(np.random.uniform(r[0], r[1], n_samples))
        return np.array(samples).T
    
    def get_fitness_range_estimate(self):
        sample = [self.get_value(x) for x in self.sample_uniform(100)]
        return np.percentile(sample, 75) - np.percentile(sample, 25), np.mean(sample)