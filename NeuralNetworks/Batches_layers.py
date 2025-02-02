# In this file we going learn the batches and implementing two layers and Objects

# Batches
# What is the need of the batches?
# If we train the model on all the data at a time, needs lots of memory.
# So, when we use the batches it will fetch the those many inputs at a time to fit the model.
# Saves memory, lesser the batch size, lesser memory consumption.


# Single sample input
import numpy as np

Batch = 8 # 16, 32, 64, 128

model_input = np.array([[1,2,3],
                        [11,21,31],
                        [12,23,33],
                        [13,24,35],
                        [14,25,36],
                        [15,26,37],
                        [16,27,38],
                        [17,28,39],
                        [18,29,30],])

# First layer with the four neurons
weight1 = np.array([[0.2, 0.4, 0.6],
           [0.15, 0.25, 0.35],
           [0.05, 0.02, 0.30],
           [0.45, 0.55, 0.35]])  # Shape

bias1 = [1,2,3,4]
print('Shape of input = ',model_input.shape)
print('Shape of weight1 = ', weight1.shape)

weight2 = np.array([[0.1, 0.2, 0.3, 0.4],
           [0.45, 0.25, 0.65, 0.23],
           [0.25, 0.72, 0.36, 0.13],
           [0.65, 0.15, 0.25, 0.12]])  # Shape

bias2 = [1.5, 2.5, 2.3, 2.1]
print('Shape of weight2 = ', weight2.shape)

layer1_output = np.dot(model_input, weight1.T) + bias1
print(layer1_output.shape)
layer2_output = np.dot(layer1_output, weight2.T) + bias2
print(layer2_output)


# # Batch input
# import numpy as np

# Batch = 8 # 16, 32, 64, 128

# model_input = np.array([[1,2,3],
#                         [11,21,31],
#                         [12,23,33],
#                         [13,24,35],
#                         [14,25,36],
#                         [15,26,37],
#                         [16,27,38],
#                         [17,28,39],
#                         [18,29,30],])

# # First layer with the four neurons
# weights = np.array([[0.2, 0.4, 0.6],
#            [0.15, 0.25, 0.35],
#            [0.05, 0.02, 0.30],
#            [0.45, 0.55, 0.35]])  # Shape

# bias = [1,2,3,4]
# print('Shape of input = ',model_input.shape)
# print('Shape of weights = ', weights.shape)

# layer1_output = np.dot(model_input, weights.T) + bias
# print(layer1_output)