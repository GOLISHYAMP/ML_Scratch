from sklearn import datasets
from sklearn.model_selection import train_test_split
from my_ML_models.LogisticRegression import LogisticRegression
import numpy as np

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=31)
lr = LogisticRegression(lr=0.0001)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy(y_pred, y_test))