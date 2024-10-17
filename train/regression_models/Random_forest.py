from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

class RandomForest_Model:
    def __init__(self, n_estimators=100, criterion='squared_error', max_features=1):
        #criteiron=absolute_error
        #max_features = "sqrt"
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_features=max_features)

    def convert_dtype(X):
        if X is None:
            return X
        X_converted = X.astype('float32')
        X_converted[np.isinf(X_converted)] = np.finfo(np.float32).max
        return X_converted

    def train(self, X_train, y_train):
        X_train, y_train = RandomForest_Model.convert_dtype(X_train), RandomForest_Model.convert_dtype(y_train)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        return np.mean([mean_squared_error(y_train[:,i], predictions[:,i]) for i in range(y_train.shape[1])]), predictions #.score(X_train, y_train)

    def test(self, X_test, y_test=None):
        X_test, y_test = RandomForest_Model.convert_dtype(X_test), RandomForest_Model.convert_dtype(y_test)
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