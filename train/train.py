
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
import time
import copy

from functions import get_base_functions,get_function_test, augment_functions
from algorithms import get_algorithms_lambdas
from fla.FLA import FLA
from performance_metrics.performance_metrics import full_comparison_oriented_scores
from regression_models.Random_forest import *


DATASET_PATH = "dataset"
SEED = 33

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
    train_functions, n1, test_functions, n2 = get_function_test(SEED)
    with open(f"{path}/function_names.txt", "w") as f:
        f.write(str(n1))
        f.write("\n\n")
        f.write(str(n2))
    train_functions = augment_functions(train_functions)
    test_functions = augment_functions(test_functions)
    print(f"Len training fun: {len(train_functions)} - Len test fun: {len(test_functions)}")
    with open(f"{path}/test_functions.pickle", "wb") as f:
        pickle.dump([train_functions, test_functions], f)

#Solve set of functions with algorithms
def solve_with_algorithm(args):
    algorithm, repeat = args
    return np.mean([algorithm.solve() for _ in range(repeat)])

def build_raw_scores(functions, budget, repeat, multi_process):
    algorithms = get_algorithms_lambdas()
    results = []
    for i, algorithm in enumerate(algorithms):
        print(f"Running algorithm {i} of {len(algorithms)}")
        if (multi_process > 1):
            with Pool(multi_process) as pool:
                results.append(pool.map(solve_with_algorithm, [(algorithm(function, budget), repeat) for function in functions]))
        else:
            results.append(list(map(solve_with_algorithm, [(algorithm(function, budget), repeat) for function in functions])))
    return results

def build_raw_scores_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_functions.pickle", "rb") as f:
        functions = pickle.load(f)
    scores = build_raw_scores(functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"], config["MULTI_PROCESS"])
    with open(f"{path}/train_scores_raw.pickle", "wb") as f:
        pickle.dump(scores, f)

def build_raw_scores_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_functions.pickle", "rb") as f:
        train_functions, test_functions = pickle.load(f)
    train_scores = build_raw_scores(train_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"], config["MULTI_PROCESS"])
    test_scores = build_raw_scores(test_functions, config["OPTIMIZATION_BUDGET"], config["REPEAT_BENCHMARK"], config["MULTI_PROCESS"])
    with open(f"{path}/test_scores_raw.pickle", "wb") as f:
        pickle.dump([train_scores, test_scores], f)

def build_scores_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_scores_raw.pickle", "rb") as f:
        raw_scores = pickle.load(f)
    transformed_scores = full_comparison_oriented_scores(raw_scores)
    with open(f"{path}/train_scores.pickle", "wb") as f:
        pickle.dump(transformed_scores, f)

def build_scores_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_scores_raw.pickle", "rb") as f:
        raw_scores_train, raw_scores_test = pickle.load(f)
    transformed_scores_train = full_comparison_oriented_scores(raw_scores_train)
    transformed_scores_test = full_comparison_oriented_scores(raw_scores_test)
    with open(f"{path}/test_scores.pickle", "wb") as f:
        pickle.dump([transformed_scores_train, transformed_scores_test], f)

#Compute and export FLA measures
def get_fla(args):
    try:
        f, config = args
        return FLA.get_FLA_measures(f, config["FLA_PARAMS"]["random_sample_N"], config["FLA_PARAMS"]["FEM_params"], config["FLA_PARAMS"]["jensens_inequality_N"], NON_DETERMINISTIC=config["NON_DETERMINISTIC"])
    except:
        return get_fla((f, config))
def compute_FLA_split(functions, config):
    if (config["MULTI_PROCESS"] > 1):
        with Pool(config["MULTI_PROCESS"]) as pool:
            return pool.map(get_fla, [(f, config) for f in functions])
    else:
        return list(map(get_fla, [(f, config) for f in functions]))

def compute_FLA_measures(functions, config, file_name):
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            fla = pickle.load(f)
    else:
        fla = []
    while len(fla)<len(functions):
        print(f"Computing split from {len(fla)} over {len(functions)}")
        fla += compute_FLA_split(functions[len(fla):min(len(functions), len(fla)+30)], config)
        with open(file_name, "wb") as f:
            pickle.dump(fla, f)
    return fla

def build_fla_measures_train(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/train_functions.pickle", "rb") as f:
        functions = pickle.load(f)
    fla = compute_FLA_measures(functions, config, f"{path}/fla.pickle")
    with open(f"{path}/fla.pickle", "wb") as f:
        pickle.dump(fla, f)
def build_fla_measures_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    with open(f"{path}/test_functions.pickle", "rb") as f:
        train_functions, test_functions = pickle.load(f)
    fla_train = compute_FLA_measures(train_functions, config, f"{path}/fla_test_temp_1.pickle")
    fla_test = compute_FLA_measures(test_functions, config, f"{path}/fla_test_temp_2.pickle")
    with open(f"{path}/fla_test.pickle", "wb") as f:
        pickle.dump([fla_train, fla_test], f)
    os.remove(f"{path}/fla_test_temp_1.pickle")
    os.remove(f"{path}/fla_test_temp_2.pickle")


#Training and testing of regression model(s)
def build_dataset(scores, fla_measures):
    #fla: [fla_measures for f in functions]
    #scores: [[score for f in functions] for a in algorithm]
    #Training dataset: [(FLA, scores for each algorithm) for each function]
    x = np.array(fla_measures)
    y = np.array([[scores[a][i] for a in range(len(scores))] for i in range(len(scores[0]))])
    return x, y
def preprocess_1(x_tr, x_te):
    x_tr = copy.deepcopy(x_tr)
    x_te = copy.deepcopy(x_te)
    x_tr = np.array([x.astype('float32') for x in x_tr])
    for x in x_tr:
        x[np.isinf(x)] = np.finfo(np.float32).max
    x_te = np.array([x.astype('float32') for x in x_te])
    for x in x_te:
        x[np.isinf(x)] = np.finfo(np.float32).max
    return x_tr, x_te

def train_and_test(experiment_name):
    config = load_config(experiment_name)
    path = f"{DATASET_PATH}/{experiment_name}"
    #Load FLA measures
    with open(f"{path}/fla_test.pickle", "rb") as f:
        fla_train, fla_test = pickle.load(f)
    #Load scores
    with open(f"{path}/test_scores.pickle", "rb") as f:
        scores_train, scores_test = pickle.load(f)
    #todo


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'function_name',
        type=str,
        choices=['makeversion', 'maketrainfunctions', 'maketestfunctions', 'maketrainrawscores', 'maketestrawscores','maketrainscores','maketestscores','makeflatest','makefla','trainandtest'],
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
    elif args.function_name == 'makefla':
        build_fla_measures_train(experiment_name)
    elif args.function_name == 'makeflatest':
        build_fla_measures_test(experiment_name)
    elif args.function_name == 'trainandtest':
        train_and_test(experiment_name)
    else:
        print(f"Unknown function: {args.function_name}")

if __name__ == "__main__":
    main()