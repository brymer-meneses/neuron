from abc import abstractmethod, ABC
from typing import Dict

from neuron.nn.layers import Layer
from neuron.nn.optimizers import Optimizer
from neuron.tensor import Tensor

class Model(ABC):

    __is_compiled = False
    __layers: Dict[str, Layer] = {}

    def __call__(self, inputs: Tensor) -> Tensor:
        assert self.__is_compiled, "Cannot propagate data to a model that is not compiled."
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def optimize(self) -> None:
        assert self.__is_compiled, "Cannot optimize a model that is not compiled."

        for layer in self.__layers.values():
            self.optimizer.step(layer)

    def compile(self, optimizer: Optimizer) -> None:
        self.__is_compiled = True

        self.optimizer = optimizer

        attributes = vars(self)
        
        for key, value in attributes.items():
            if isinstance(value, Layer):
                self.__layers[key] = value
                value.name = key

    def summary(self) -> None:
        # TODO: improve this
        for name, layer in self.__layers.items():
            print(f"{name} - {layer}")

        print(f"Optimizer: {self.optimizer}")
        return

    def zero_grad(self) -> None:
        for layer in self.__layers.values():
            layer.zero_grad()

