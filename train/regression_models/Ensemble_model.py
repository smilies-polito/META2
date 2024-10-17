from meta_optimization_model.SVR import *
from meta_optimization_model.Random_forest import *
import copy 
import os

class EnsembleModel:
    def __init__(self,DEBUG_LOGS = False):
        self.models = [
                SVR_Model(kernel="rbf"),
                SVR_Model(kernel="poly"),
                RandomForest_Model(n_estimators=100, criterion='squared_error', max_features="sqrt"),
                RandomForest_Model(n_estimators=50, criterion='squared_error', max_features="sqrt"),
                RandomForest_Model(n_estimators=150, criterion='squared_error', max_features="sqrt")
            ]
        self.ensemble_model = RandomForest_Model(max_features=None)
        self.DEBUG_LOGS = DEBUG_LOGS

    def train(self, X_train, y_train):
        predicted = np.zeros((X_train.shape[0], 0))
        for i, model in enumerate(self.models):
            X_train_copy, y_train_copy = copy.deepcopy(X_train), copy.deepcopy(y_train) 
            errors, pred = model.train(X_train_copy, y_train_copy)
            predicted = np.append(predicted, pred, axis=1)
            if (self.DEBUG_LOGS):
                print(f"Model {i}, error is {errors}")
        predicted = np.array(predicted)
        return self.ensemble_model.train(predicted, y_train)
        
    def test(self, X_test, y_test=None):
        predicted = np.zeros((X_test.shape[0], 0))
        for model in self.models:
            predicted = np.append(predicted, model.test(X_test, y_test)[0], axis=1)
        return self.ensemble_model.test(predicted, y_test)

    def export_model(self, path):
        os.makedirs(path, exist_ok = True)
        for i, model in enumerate(self.models):
            model.export_model(os.path.join(path, f"model_{i}"))
        self.ensemble_model.export_model(os.path.join(path, "ensemble"))
    
    def load_model(path):
        assert os.path.isdir(path)
        model = EnsembleModel()
        model.models = [
            SVR_Model.load_model(os.path.join(path, f"model_0")),
            SVR_Model.load_model(os.path.join(path, f"model_1")),
            RandomForest_Model.load_model(os.path.join(path, f"model_2")),
            RandomForest_Model.load_model(os.path.join(path, f"model_3")),
            RandomForest_Model.load_model(os.path.join(path, f"model_4")),
        ]
        model.ensemble_model = RandomForest_Model.load_model(os.path.join(path, f"ensemble"))
        return model
    
    """def export_model(self, path):
        assert self.models is not None
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)"""