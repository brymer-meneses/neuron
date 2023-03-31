
from abc import abstractmethod

from neuron.nn.layers import Layer

class Optimizer:
    @abstractmethod
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class SGG(Optimizer):

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, layer: Layer) -> None:

        assert layer.weights.grad is not None, "Cannot optimize layer with no gradients"
        
        layer.weights.data = layer.weights.data - self.lr * layer.weights.grad

        if layer.bias:
            assert layer.bias.grad is not None, "Cannot optimize layer with no gradients"

            layer.bias.data = layer.bias.data - self.lr * layer.bias.grad




