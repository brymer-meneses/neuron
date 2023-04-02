
from abc import abstractmethod

from neuron.nn.layers import Layer

class Optimizer:
    @abstractmethod
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, layer: Layer) -> None:

        for param in layer.params.values():
            assert param.grad is not None, "Cannot optimize layer with no gradients"

            param.data = param.data - self.lr * param.grad
