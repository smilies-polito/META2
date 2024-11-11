import sys 
sys.path.append("../")
from functions import get_base_functions, get_split_functions, augment_functions
import matplotlib.pyplot as plt
import os
import random
from problems.Multiplied_problems import *
from problems.Summed_problems import *

def plot_function(problem, path, resolution=100):
    x = np.arange(problem.get_ranges()[0][0], problem.get_ranges()[0][1], (problem.get_ranges()[0][1]-problem.get_ranges()[0][0])/resolution)
    y = np.arange(problem.get_ranges()[1][0], problem.get_ranges()[1][1], (problem.get_ranges()[1][1]-problem.get_ranges()[1][0])/resolution)
    z = [[problem.get_value(np.array([i,j])) for i in x] for j in y]
    plt.imshow(z, cmap='hot', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.savefig(path+".png")

functions = get_base_functions()
functions = [f for f in functions if f.get_dimensions()==2]
print(len(functions))

for i in range(10):
    os.makedirs(f"junk_to_delete/{i}", exist_ok=True)
    f1, f2 = random.choice(functions), random.choice(functions)
    s = Summed_problems(f1, f2, weight_1=random.random()*0.7+0.15)
    m = Multiplied_problems(f1, f2)
    plot_function(f1, f"junk_to_delete/{i}/f1")
    plot_function(f2, f"junk_to_delete/{i}/f2")
    plot_function(s, f"junk_to_delete/{i}/s")
    plot_function(m, f"junk_to_delete/{i}/m")