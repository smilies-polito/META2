from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 
import numpy as np
import pickle

class MLP_Model:
    def __init__(self, alpha=0.0001,hidden_layer_sizes=(100,),max_iter=200,convert_dtype=False):
        #alpha: between 0.001 and 0.8
        #Hidden layer sizes; (30,30,) ..?
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,alpha=alpha,max_iter=max_iter)
        self.mean = None
        self.convert_dtype=convert_dtype

    def run_convert_dtype(self,X):
        if X is None or self.convert_dtype==False:
            return X
        X = np.array([x.astype('float32') for x in X])
        for x in X:
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 0
        return X
    
    def normalize_features(self,X):
        if self.convert_dtype==False:
            return X
        if (self.mean is None):
            self.mean = np.mean(X, 0)
            self.min = np.min(X, 0)
            self.max = np.max(X, 0)
        return (X-self.mean)/(self.max-self.min)

    def train(self, X_train, y_train):
        X_train = self.normalize_features(X_train)
        X_train, y_train = self.run_convert_dtype(X_train), self.run_convert_dtype(y_train)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        return np.mean([mean_squared_error(y_train[:,i], predictions[:,i]) for i in range(y_train.shape[1])]), predictions #.score(X_train, y_train)

    def test(self, X_test, y_test=None):
        X_test = self.normalize_features(X_test)
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