import numpy as np
from sklearn.datasets import make_classification # type: ignore

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, LayerInput):
        self.LayerOutput = np.dot(LayerInput, self.weights.T) + self.bias

class ActivationRelu:
    def forward(self, LayerOutput):
        self.ActivationOutput = np.maximum(0, LayerOutput)

class ActivationSoftmax:
    def forward(self, LayerOutput):
        exp_output = np.exp(LayerOutput)
        self.ActivationOutput = exp_output / np.sum(exp_output, axis = 1, keepdims= True)

class Loss:
    def Calculate(self, X, Y):
        Preds = self.CategoricalCrossEntropy(X, Y)
        print("Accuracy : ",Preds)
 

    def CategoricalCrossEntropy(self, ActivationOutput, y):
        n_sample = len(y)
        self.ActivationOutput = np.clip(ActivationOutput, 1e-7, 1-1e-7)
        self.ActivationOutput = -np.log(self.ActivationOutput)
        if len(y.shape) == 1:
            # It means that the Y is the direct labels eg: [1, 0, 2]
            self.ActivationOutput = self.ActivationOutput[np.arange(n_sample), y]
            return np.mean(self.ActivationOutput)
        else:
            # It means that the Y is the One hot encoding
            self.ActivationOutput = -(np.log(self.ActivationOutput))
            # output = output[np.arange(3), target]
            self.ActivationOutput = self.ActivationOutput * y
            self.ActivationOutput = np.sum(self.ActivationOutput, axis= 1, keepdims=True)
            acc = np.mean(self.ActivationOutput)
            print('Accuracy : ', acc)

        

X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, n_clusters_per_class=1 )
print(X.shape, y.shape)
L1 = DenseLayer(8, 4)  # Number of neurons and input shape of single row
L2 = DenseLayer(4, 8)

Relu = ActivationRelu()
Softmax = ActivationSoftmax()

L1.forward(X)
Relu.forward(L1.LayerOutput)
L2.forward(Relu.ActivationOutput)
Softmax.forward(L2.LayerOutput)
print(Softmax.ActivationOutput.shape)

loss = Loss()
loss.Calculate(Softmax.ActivationOutput, y)
