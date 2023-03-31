from abc import abstractmethod
from typing import Optional

from neuron import Tensor

class Layer:

    weights: Tensor
    bias: Optional[Tensor]

    @abstractmethod
    def __init__(self, in_features: int, out_features: int, use_bias: bool, name: str) -> None:
        pass

    @abstractmethod
    def forward(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, use_bias: bool, name: str = 'Linear') -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.name = name

        self.weights = Tensor.random((out_features, in_features), requires_grad=True)
        
        if self.use_bias:
            self.bias = Tensor.random((out_features, 1), requires_grad=True)
        else:
            self.bias = None

    def __repr__(self) -> str:
        return f"<Linear in_features={self.in_features} out_features={self.out_features} use_bias={self.use_bias}>"

    def forward(self, data: Tensor) -> Tensor:
        assert self.weights.shape[1] == data.shape[0], f"Cannot forward data with shape {data.shape} to Linear Layer with shape {self.weights.shape}."

        if self.bias:
            self.output = self.weights @ data + self.bias
        else:
            self.output = self.weights @ data

        return self.output

    def zero_grad(self) -> None:
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()
        return


