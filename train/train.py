
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
import argparse
import json 
import os
import pickle
from multiprocessing import Pool
import numpy as np

from functions import get_base_functions, get_split_functions, augment_functions
from algorithms import get_algorithms_lambdas

DATASET_PATH = "dataset"
#New experiment. Experiment name as argument. Copy default config is config file not existing
def new_experiment(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    #Create directory, create config.json inside directory
    os.makedirs(path, exist_ok=True)
    #No config.json, copy base_config.json
    if not os.path.exists(f"{path}/config.json"):
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
def define_test_function_set(experiment_name):
    path = f"{DATASET_PATH}/{experiment_name}"
    train_functions, test_functions = get_split_functions()
    train_functions = augment_functions(train_functions)
    test_functions = augment_functions(test_functions)
    with open(f"{path}/test_functions.pickle", "wb") as f:
        pickle.dump([train_functions, test_functions], f)

#Solve set of functions with algorithms
def solve_with_algorithm(args):
    algorithm, repeat = args
    return np.mean([algorithm.solve() for _ in range(repeat)])

def build_raw_scores(functions, budget, repeat):
    algorithms = get_algorithms_lambdas()
    results = []
    for i, algorithm in enumerate(algorithms):
        print(f"Running algorithm {i} of {len(algorithms)}")
        with Pool(8) as pool:
            results.append(pool.map(solve_with_algorithm, [(algorithm(function, budget), repeat) for function in functions]))


def build_raw_scores_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_functions.pickle", "rb") as f:
        functions = pickle.load(f)
    scores = build_raw_scores(functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    with open(f"{path}/train_scores_raw.pickle", "wb") as f:
        pickle.dump(scores, f)

def build_raw_scores_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_functions.pickle", "rb") as f:
        train_functions, test_functions = pickle.load(f)
    train_scores = build_raw_scores(train_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    test_scores = build_raw_scores(test_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"])
    with open(f"{path}/test_scores_raw.pickle", "wb") as f:
        pickle.dump([train_scores, test_scores], f)

def build_scores_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_scores_raw.pickle", "rb") as f:
        raw_scores = pickle.load(f)
    transformed_scores = full_comparison_oriented_scores(raw_scores)
    with open(f"{path}/train_scores.pickle", "rb") as f:
        pickle.dump(transformed_scores, f)

def build_scores_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_scores_raw.pickle", "rb") as f:
        raw_scores_train, raw_scores_test = pickle.load(f)
    transformed_scores_train = full_comparison_oriented_scores(raw_scores_train)
    transformed_scores_test = full_comparison_oriented_scores(raw_scores_test)
    with open(f"{path}/test_scores.pickle", "rb") as f:
        pickle.dump(transformed_scores, f)

#Compute and export FLA measures
def build_fla_measures_train(experiment_name):
    pass
def build_fla_measures_test(experiment_name):
    pass



def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'function_name',
        type=str,
        choices=['makeversion', 'maketrainfunctions', 'maketestfunctions', 'maketrainrawscores', 'maketestrawscores','maketrainscores','maketestscores'],
        help="Using this file by hand is not recommended. Use the makefile to train the model"
    )
    parser.add_argument(
        'experiment_name',
        type=str,
    )

    # Parse the arguments
    args = parser.parse_args()
    experiment_name = args.experiment_name
    # Call the selected function
    if args.function_name == 'makeversion':
        new_experiment(experiment_name)
    elif args.function_name == 'maketrainfunctions':
        define_train_function_set(experiment_name)
    elif args.function_name == 'maketestfunctions':
        define_test_function_set(experiment_name)
    elif args.function_name == 'maketrainrawscores':
        build_raw_scores_train(experiment_name)
    elif args.function_name == 'maketestrawscores':
        build_raw_scores_test(experiment_name)
    elif args.function_name == 'maketrainscores':
        build_scores_train(experiment_name)
    elif args.function_name == 'maketestscores':
        build_scores_test(experiment_name)
    else:
        print(f"Unknown function: {args.function_name}")

if __name__ == "__main__":
    main()