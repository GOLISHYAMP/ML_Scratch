# In this file we are going to implement the neural network 
# to solve the classification problem along with the activation function.

import numpy as np
from sklearn.datasets import make_classification # type: ignore

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, layerInput):
        self.layerOutput = np.dot(layerInput, self.weights.T) + self.bias
        return self.layerOutput
    
class ActivationRelu:
    def forward(self, output):
        self.ActivationOutput = np.maximum(0, output)

X, y = make_classification(n_samples= 1000, n_features= 4, n_classes=2)
print(X.shape)
print(y.shape)
print(X[0])
L1 = DenseLayer(len(X[0]), 8)
L2 = DenseLayer(8, 8)
Layer1_Output = L1.forward(X)
# print(Layer1_Output)
Relu = ActivationRelu()
Relu.forward(Layer1_Output)
# print(Relu.ActivationOutput)
Layer2_Output = L2.forward(Relu.ActivationOutput)
Relu.forward(Layer2_Output)
print(Relu.ActivationOutput)
print(Relu.ActivationOutput.shape)