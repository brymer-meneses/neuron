from abc import abstractmethod
from typing import List

from neuron.nn.layers import Layer
from neuron.nn.optimizers import Optimizer
from neuron.nn.loss import Loss
from neuron.tensor import Tensor


class Model:

    __is_compiled = False
    __layers: List[Layer] = []

    def __call__(self, inputs: Tensor) -> Tensor:
        assert self.__is_compiled, "Cannot propagate data to a model that is not compiled."
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def optimize(self) -> None:
        assert self.__is_compiled, "Cannot optimize a model that is not compiled."

        for layer in self.__layers:
            self.optimizer.step(layer)

    def compile(self, optimizer: Optimizer, loss: Loss) -> None:
        self.__is_compiled = True

        self.optimizer = optimizer
        self.loss = loss
        
        for _, value in vars(self):
            if isinstance(value, Layer):
                self.__layers.append(value)

    def zero_grad(self) -> None:
        for layer in self.__layers:
            layer.zero_grad()

