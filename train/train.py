
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
from functions import get_base_functions, get_split_functions, augment_functions

#New experiment. Experiment name as argument. Config json as stdin, or default config if nothing
def new_experiment(experiment_name):
    #Verify experiment not already existing
    #Create directory, create config.json inside directory
    pass

#Load config, return dictionary
def load_config(experiment_name):
    pass

#Define set of functions to train the model
def define_train_function_set(experiment_name):
    functions = get_base_functions()
    augment_functions(functions)
    #todo export

#Define two sets of functions to train and validate the model
def define_train_function_set(experiment_name):
    train_functions, test_functions = get_split_functions()
    train_functions = augment_functions(train_functions)
    test_functions = augment_functions(test_functions)
    #todo export

#Solve set of functions with algorithms - export scores in csv or json
def build_scores(experiment_name):
    pass 

#Compute and export FLA measures
def build_fla_measures(experiment_name):
    pass


