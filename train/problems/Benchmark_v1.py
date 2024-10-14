import numpy as np
import math
from noise import snoise2
from problems.BaseProblem import BaseProblem
class Benchmark_v1(BaseProblem):
    #First component: given hyperplane defined by y = 100 + x, return distance of point (x,y) from hyperplane
    linear_hyperplane_distance = lambda x, y: abs(y - 100 - x)
    #Second component: distance of x from point 800 on x-axis
    distance_from_point = lambda x, y: abs(x - 800)
    #Third component: sin(z) where z is the projection of (x,y) on the line y = 100+x divided by the sqrt of distance from the hyperplane
    project_over_hyperplane = lambda x, y: (x + y) / 2
    sin_projection = lambda x, y: abs(math.sin(Benchmark_v1.project_over_hyperplane(x,y)/30))*50 / 1 + (((Benchmark_v1.linear_hyperplane_distance(x, y)+1))*0.4)
    #Fourth component: noise
    def noise(x, y, scale=200, octaves=3, persistence=4, lacunarity=0.0001):
        return snoise2(x * scale, y * scale, octaves=octaves, persistence=persistence,lacunarity=lacunarity)

    def __init__(self, linear_hyperplane_distance_weight=0.7, distance_from_point_weight=0.2, sin_projection_weight=0.5, noise_weight=80, f_range=((0, 6000), (0, 6000))):
        self.linear_hyperplane_distance_weight = linear_hyperplane_distance_weight
        self.distance_from_point_weight = distance_from_point_weight
        self.sin_projection_weight = sin_projection_weight
        self.noise_weight = noise_weight
        self.ranges = f_range
        self.dimensions = 2
        self.name = f"Benchmark_v1_2d"

    def get_value(self, vector):
        x, y = vector[0], vector[1]
        return Benchmark_v1.linear_hyperplane_distance(x,y)*self.linear_hyperplane_distance_weight + Benchmark_v1.distance_from_point(x,y)*self.distance_from_point_weight + Benchmark_v1.sin_projection(x,y)*self.sin_projection_weight + Benchmark_v1.noise(x,y)*self.noise_weight

