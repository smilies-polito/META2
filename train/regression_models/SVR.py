from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

class SVR_Model:
    def __init__(self, kernel="linear"):
        # Create the SVR model with a linear kernel
        self.kernel = kernel
        self.std__ = None; self.mean__ = None
        self.models = None

    def convert_dtype(X):
        if X is None:
            return X
        X = np.array([x.astype('float32') for x in X])
        for x in X:
            x[np.isinf(x)] = np.finfo(np.float32).max
        return X

    def normalize_input(self, X):
        if (self.std__ is None):
            std__ =  np.std(X, axis=0)
            std__[np.isinf(std__)] = np.finfo(np.float64).max
            std__[std__==0] = 1
            self.std__ = std__
        if (self.mean__ is None):
            mean__ = np.mean(X, axis=0)
            mean__[np.isinf(mean__)] = np.finfo(np.float64).max
            self.mean__ = mean__
        return (X - self.mean__) / self.std__

    def train(self, X_train, y_train):
        X_train = self.normalize_input(X_train)
        self.models = [SVR(kernel=self.kernel) for _ in range(y_train.shape[1])]
        predictions = np.zeros(y_train.shape)
        for i in range(len(self.models)):
            self.models[i].fit(X_train, y_train[:, i])
            predictions[:,i] = self.models[i].predict(X_train)
        return np.mean([mean_squared_error(y_train[:,i], predictions[:,i]) for i in range(y_train.shape[1])]), predictions
        

    def test(self, X_test, y_test=None):
        assert self.models is not None
        X_test = self.normalize_input(X_test)
        X_test[X_test>100]=100
        X_test[X_test<-100]=-100
        predictions = np.array([model.predict(X_test) for model in self.models]).T
        if np.isnan(predictions).any():
            print(len(predictions[np.isnan(predictions)]))
            assert False 
        if y_test is None:
            return predictions, None
        errors = [mean_squared_error(y_test[:, i], predictions[:, i]) for i in range(len(self.models))]
        return predictions, errors

    def export_model(self, path):
        assert self.models is not None
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)