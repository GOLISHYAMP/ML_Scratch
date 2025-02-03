import numpy as np
layerInput = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(layerInput.max(axis=1, keepdims=True))
# Here we subtracting the max of the layer with others so that upper bond will not reach.
layerInput = layerInput - layerInput.max(axis=1, keepdims=True)
print(layerInput)
exp_input = np.exp(layerInput)
print(exp_input)
print(np.sum(exp_input, axis=1, keepdims=True))
output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
print(output)



