
import numpy as np

class LinearRegression:
    def __init__(self, n_iters=1000):
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y, lr = 0.01):
        data_size , n_features = X.shape
        print(X.shape)
        print(y.shape)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # dw = (1/data_size)* np.dot(X.reshape(n_features,-1), (y_pred - y))
            dw = (1/data_size)* np.dot(X.T, (y_pred - y))
            db = (1/data_size) * np.sum(y_pred - y)

            self.weights = self.weights - lr * dw
            self.bias = self.bias - lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def params(self):
        return (self.weights, self.bias)