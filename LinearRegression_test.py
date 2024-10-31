from my_ML_models.LinearRegression import LinearRegression

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def mse(y_pred, y):
    return np.mean((y_pred - y)**2)

X, y = datasets.make_regression(n_samples=500, n_features=1, noise= 50, random_state= 31)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 31)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("Accuracy = ", mse(y_pred, y_test))

fig = plt.figure(figsize=(8, 6))
plt.scatter(x = X_train[:, 0], y=y_train, color = 'green')
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'red', label='prediction')
plt.show()
# print(X[:, :5])
# print(X_train[:, 0])
# print(X)
