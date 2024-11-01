import numpy as np

class OLS:
    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, X, y):
        # y = mx + c
        # c = mean(y) - m * mean(x)
        # m = sum(y - mean(y)) / sum(x - mean(x))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.m = np.sum(y - np.mean(y)) / np.sum(X - np.mean(X.T, axis= 1), axis= 0)
        self.c = np.mean(y) - np.dot(self.m , np.mean(X.T, axis= 1))

        # print(np.sum(y - np.mean(y)))

    def predict(self, X):
        return np.sum(np.dot(X, self.m)) + self.c
    
    def params(self):
        return (self.m, self.c)