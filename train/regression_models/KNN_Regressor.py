from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

class KNN_Model:
    def __init__(self, n_neighbors=4, weights="distance", convert_dtype=False):
        #Weights: "uniform", "distance"
        self.model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        self.convert_dtype = convert_dtype

    def run_convert_dtype(self,X):
        if X is None or self.convert_dtype==False:
            return X
        X = np.array([x.astype('float32') for x in X])
        for x in X:
            x[np.isinf(x)] = np.finfo(np.float32).max
        return X

    def train(self, X_train, y_train):
        X_train, y_train = self.run_convert_dtype(X_train), self.run_convert_dtype(y_train)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        return np.mean([mean_squared_error(y_train[:,i], predictions[:,i]) for i in range(y_train.shape[1])]), predictions 

    def test(self, X_test, y_test=None):
        X_test, y_test = self.run_convert_dtype(X_test), self.run_convert_dtype(y_test)
        predictions = self.model.predict(X_test)
        if y_test is None:
            return predictions, None
        errors = [mean_squared_error(y_test[:,i], predictions[:,i]) for i in range(y_test.shape[1])]
        return predictions, errors

    def export_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)