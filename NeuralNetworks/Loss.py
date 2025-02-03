# In this file we are going to find the loss
import numpy as np

pred = np.array([[0.7, 0.2, 0.1],
                 [0.15, 0.20, 0.65],
                 [0.1, 0.6, 0.3]])

target = np.array([0, 2, 1])   #Labels

# negative log
output = -(np.log(pred))

output = output[np.arange(3), target]
print(output)
acc = np.mean(output)
print('Accuracy : ', acc)

# target = np.array([[1, 0, 0],
#                    [0, 0, 1],
#                    [0, 1, 0]])   #One Hot Encoding
# # negative log
# output = -(np.log(pred))
# # output = output[np.arange(3), target]
# output = output * target
# output = np.sum(output, axis= 1, keepdims=True)
# print(output)
# acc = np.mean(output)
# print('Accuracy : ', acc)