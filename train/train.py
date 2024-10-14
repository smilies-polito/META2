
"""
    Training steps:
        1-Load configuration - includes directory for outputs
        2-Define set of training functions - export with pickle
        3-Produce scores for pairs (function, algorithm)
        4-Produce FLA measures for functions
        5-Train and export model
    Validation steps:
        >As training, but defining training functions require splitting original functions into two sets
        and perform data augmentation separately
"""
import json 
import os
import pickle
from multiprocessing import Pool

from functions import get_base_functions, get_split_functions, augment_functions
from algorithms import get_algorithms_lambdas

DATASET_PATH = "dataset"
#New experiment. Experiment name as argument. Copy default config is config file not existing
def new_experiment(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    #Create directory, create config.json inside directory
    os.path.makedirs(path, exist_ok=True)
    #No config.json, copy base_config.json
    if not os.path.exists(f"path/{config.json}"):
        with open(f"{DATASET_PATH}/base_config.json", "r") as f:
            config = json.load(f)
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

#Load config, return dictionary
def load_config(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/config.json", "r") as f:
        config = json.load(f)
    return config

#Define set of functions to train the model
def define_train_function_set(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    functions = get_base_functions()
    functions = augment_functions(functions)
    with open(f"{path}/train_functions.pickle", "wb") as f:
            pickle.dump(functions, f)

#Define two sets of functions to train and validate the model
def define_train_function_set(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    train_functions, test_functions = get_split_functions()
    train_functions = augment_functions(train_functions)
    test_functions = augment_functions(test_functions)
    with open(f"{path}/test_functions.pickle", "wb") as f:
        pickle.dump([train_functions, test_functions], f)

#Solve set of functions with algorithms
def solve_with_algorithm(args):
    algorithm, function, budget, repeat = args
    return np.mean([algorithm(function, budget).solve() for _ in range(repeat)])

def build_scores(functions, budget, repeat):
    algorithms = get_algorithms_lambdas()
    results = []
    for algorithm in algorithms:
        with Pool(8) as pool:
            results.append(pool.map(solve_with_algorithm, [(algorithm, function, budget, repeat) for function in functions]))

def build_scores_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_functions.pickle", "rb") as f:
        functions = pickle.load(f)
    scores = build_scores(functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    with open(f"{path}/train_scores.pickle", "wb") as f:
        pickle.dump(scores, f)

def build_scores_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_functions.pickle", "rb") as f:
        train_functions, test_functions = pickle.load(f)
    train_scores = build_scores(train_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    test_scores = build_scores(test_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    with open(f"{path}/test_scores.pickle", "wb") as f:
        pickle.dump([train_scores, test_scores], f)

#Compute and export FLA measures
def build_fla_measures_train(experiment_name):
    pass
def build_fla_measures_test(experiment_name):
    pass


