from abc import abstractmethod, ABC

from typing import Dict

from neuron import Tensor

class Layer(ABC):

    params: Dict[str, Tensor]

    @abstractmethod
    def __init__(self, in_features: int, out_features: int, use_bias: bool, name: str) -> None:
        pass

    @abstractmethod
    def forward(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, use_bias: bool=False, name: str = 'Linear') -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.name = name

        self.params: Dict[str, Tensor] = {}

        self.params["W"] = Tensor.random((out_features, in_features), requires_grad=True)
        
        if self.use_bias:
            self.params["b"] = Tensor.random((out_features, 1), requires_grad=True)

    def __repr__(self) -> str:
        return f"<Linear in_features={self.in_features} out_features={self.out_features} use_bias={self.use_bias}>"

    def forward(self, data: Tensor) -> Tensor:
        assert self.params["W"].shape[1] == data.shape[0], f"Cannot forward data with shape {data.shape} to Linear Layer with shape {self.params['W'].shape}."

        if self.use_bias:
            output = self.params["W"] @ data + self.params["b"]
        else:
            output = self.params["W"] @ data

        return output

    def zero_grad(self) -> None:

        for weight in self.params.values():
            weight.zero_grad()

        return


