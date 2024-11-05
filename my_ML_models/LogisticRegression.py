import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr = 0.001):
        self.weights = None
        self.bias = None
        self.lr = lr

    def fit(self, X, y, n_iters = 1000):
        n_samples, n_features = X.shape
        self.bias = 0
        self.weights = np.zeros(n_features)
        for _ in range(n_iters):
            LinearCal = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(LinearCal)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def params(self):
        return self.weights, self.bias
    
    def predict(self, X):
        linearCal = np.dot(X, self.weights) + self.bias
        prediction = sigmoid(linearCal)
        results = [1 if y_pred >=0.5 else 0 for y_pred in prediction]
        return results

