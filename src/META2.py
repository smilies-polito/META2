import pickle 
import json
import os
import numpy as np 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fla.FLA import FLA

class META2:
    def __init__(self, version):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset", version)
        with open(os.path.join(path,"model.pickle"),"rb") as f:
            self.regression_model = pickle.load(f)
        with open(os.path.join(path,"config.json")) as f:
            self.config = json.load(f)
    
    def predict(self, problem):
        fla = FLA.get_FLA_measures(problem, self.config["FLA_PARAMS"]["random_sample_N"], self.config["FLA_PARAMS"]["FEM_params"], self.config["FLA_PARAMS"]["jensens_inequality_N"], NON_DETERMINISTIC=False)
        predictions, _ = self.regression_model.test([fla])
        return predictions[0]

    def get_config(self):
        return config 
    
    def set_config(self, config):
        self.config = config