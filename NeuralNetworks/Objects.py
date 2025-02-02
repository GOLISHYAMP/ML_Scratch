# In this file, I have implemented the Neural Network with two Dense Layer
import numpy as np

X = [[1,2,3],
               [2,3,4],
               [5,6,7]]

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.bias = np.zeros((1,n_neurons))

    def forward(self, layer_input):
        layer_output = np.dot(layer_input, self.weights.T) + self.bias
        return layer_output
    
L1 = DenseLayer(len(X[0]), 4)
L2 = DenseLayer(4, 2)   # L1 will be the input to the L2

output_L1 = L1.forward(X)
# print(output_L1)
output_L2 = L2.forward(output_L1)
print(output_L2)
