# Fix import errors
import sys
sys.path.append("../neuron/")

from neuron.datasets import MNIST, BatchIterator
from neuron.datasets.utils import encode_one_hot

from neuron.tensor import Tensor

from neuron.nn.layers import Linear
from neuron.nn.optimizers import SGD
from neuron.nn.activations import ReLU, TanH
from neuron.nn.loss import MeanSquaredError
from neuron.nn.model import Model

import numpy as np

np.set_printoptions(threshold=False)

class LinearModel(Model):
    
    def __init__(self) -> None:
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 64)
        self.linear4 = Linear(64, 10)
        return

    def forward(self, inputs: Tensor) -> Tensor:
        outL1 = ReLU(self.linear1(inputs))
        outL2 = ReLU(self.linear2(outL1))
        outL3 = ReLU(self.linear3(outL2))
        outL4 = ReLU(self.linear4(outL3))
        return TanH(outL4)

model = LinearModel()

optimizer = SGDM(lr=0.01)
model.compile(optimizer)

model.summary()


data = MNIST(path="mnist_dataset").load()
(x_train, y_train), (x_test, y_test) = data

# normalize to avoid overflow
x_train = x_train / 255.0
x_test = x_test / 255.0

batch_iterator = BatchIterator(256)

max_epochs = 100

for epoch in range(max_epochs):

    epoch_loss = 0
    for (inputs, targets) in batch_iterator(x_train, y_train):

        inputs = inputs.reshape((784, -1))
        targets = encode_one_hot(10, targets).T

        predictions = model(inputs)

        loss = MeanSquaredError(targets, predictions)
        loss.backward()
    
        model.optimize()
        model.zero_grad()
        epoch_loss += loss.data
    
    print(f"Epoch {epoch}: {epoch_loss}")
    
