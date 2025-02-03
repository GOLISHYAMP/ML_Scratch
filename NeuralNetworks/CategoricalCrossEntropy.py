# In this file we are going to implement the Categorical Cross Entropy Loss 

import math
import numpy as np

output = [0.7, 0.2, 0.1]
target = [1, 0, 0]

# lets calculate the loss
# Loss = -(sum(y * log(yhat)))

Loss = -(target[0] * math.log(output[0]) + target[1] * math.log(output[1]) + target[2] * math.log(output[2]))
print(Loss)