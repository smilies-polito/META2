from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class Linear_Model:
    def __init__(self):
        #criteiron=absolute_error
        #max_features = "sqrt"
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        return np.mean([mean_squared_error(y_train[:,i], predictions[:,i]) for i in range(y_train.shape[1])]) #.score(X_train, y_train)


    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        errors = [mean_squared_error(y_test[:,i], predictions[:,i]) for i in range(y_test.shape[1])]
        return predictions, errors