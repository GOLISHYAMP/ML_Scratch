import numpy as np

layer_input = np.array([1,2,3])

weight = np.array([[0.1, 0.2, 0.2],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
            [0.4, 0.5, 0.6]])

bias = np.array([0.1, 0.2, 0.3, 0.4])

print(layer_input.shape)
print(weight.shape)

output = np.dot(weight, layer_input) + bias
print(output)