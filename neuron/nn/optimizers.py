
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

class SGDM(Optimizer):
    """Stochastic Gradient Descent with Momentum"""

    def __init__(self, lr: float=0.01, beta: float=0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.V: Dict[str, np.ndarray] = {}

    def step(self, layer: Layer) -> None:

        assert layer.name is not None

        for param_name, param in layer.params.items():

            assert param.grad is not None

            name = f"{layer.name}[{param_name}]"

            self.V[name] = self.V.get(name, np.array(0))
            self.V[name] = self.beta * self.V[name] + (1 - self.beta) *  param.grad

            param.data = param.data - self.lr * self.V[name]

