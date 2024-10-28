
# In KNN, here the training will not be happened, in training only the datasets is stored.
# During the prediction only all the calculations will be done. that why training is quick in KNN
# But inference takes longer time.

import numpy as np
import math
from collections import Counter

def euclidean_distance(x1, x2):
    return math.sqrt( np.sum((x2-x1)**2))  # (x2 -x1)**2 + (y2 -y1)**2 + (z2 -z1)**2

class KNN:
    def __init__(self, neighbor = 3) -> None:
        self.k = neighbor

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x) -> int:
        # calculate the distance with each of the datapoints using euclidean method
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # now get the k nearest points index numbers
        indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in indexes]
        
        # find the majority
        most_common = Counter(k_nearest_labels).most_common()
        return int(most_common[0][0])