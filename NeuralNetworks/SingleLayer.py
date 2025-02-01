# In this we going to implement a layer with four neurons, with three inputs
# import numpy as np

# layer_input = [1, 2, 3]   # shape  

# # four neurons, so four different set of weights
# weight1 = [0.1, 0.2, 0.2]  # shape
# weight2 = [0.2, 0.4, 0.6]
# weight3 = [0.3, 0.6, 0.9]
# weight4 = [0.4, 0.5, 0.6]

# bias1 = 0.2
# bias2 = 0.3
# bias3 = 0.4
# bias4 = 0.5

# output_neuron1 = layer_input[0]*weight1[0] + layer_input[1]*weight1[1] + layer_input[2]*weight1[2] + bias1
# output_neuron2 = layer_input[0]*weight2[0] + layer_input[1]*weight2[1] + layer_input[2]*weight2[2] + bias2
# output_neuron3 = layer_input[0]*weight3[0] + layer_input[1]*weight3[1] + layer_input[2]*weight3[2] + bias3
# output_neuron4 = layer_input[0]*weight4[0] + layer_input[1]*weight4[1] + layer_input[2]*weight4[2] + bias4

# print([output_neuron1, output_neuron2, output_neuron3, output_neuron4])



import numpy as np

layer_input = np.array([1, 2, 3])   # shape  

# four neurons, so four different set of weights
weight1 = np.array([0.1, 0.2, 0.2])  # shape
weight2 = np.array([0.2, 0.4, 0.6])
weight3 = np.array([0.3, 0.6, 0.9])
weight4 = np.array([0.4, 0.5, 0.6])

bias1 = 0.2
bias2 = 0.3
bias3 = 0.4
bias4 = 0.5

output_neuron1 =  np.sum(layer_input * weight1)+ bias1
output_neuron2 =  np.sum(layer_input * weight2)+ bias2
output_neuron3 =  np.sum(layer_input * weight3) + bias3
output_neuron4 =  np.sum(layer_input * weight4) + bias4

print([output_neuron1, output_neuron2, output_neuron3, output_neuron4])