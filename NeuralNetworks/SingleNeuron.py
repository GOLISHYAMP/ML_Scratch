import numpy as np
input = [1,2,3]
weights = [0.1, 0.2, 0.3]
bias = 0.1

output = (input[0] * weights[0]) +(input[1] * weights[1]) +(input[2] * weights[2]) + bias
sigmoid = 1/ (1 + np.exp(-output))
print(sigmoid)