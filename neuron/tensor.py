
from typing import Callable, Optional, Tuple, Union, NamedTuple, List
import numpy as np

Numeric = Union[float, List, np.ndarray, 'Tensor']

class Dependency(NamedTuple):
    tensor: 'Tensor'
    gradfn: Callable[[np.ndarray], np.ndarray]
    gradfn_name: str

class Tensor:

    def __init__(self, data: Numeric, requires_grad: bool = False) -> None:

        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.shape = self.data.shape

        self.depends_on: List[Dependency] = []
        self.grad: Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

        return

    def backward(self, grad: Optional[Numeric] = None) -> None:
        assert self.requires_grad, "Cannot call `backwards` to a tensor that has `requires_grad=False.`"

        if grad is None:
            if self.shape == ():
                grad = np.array(1)
            else:
                raise RuntimeError("Gradient for non-scalar tensor must be specified.")

        grad = _ensure_array(grad)

        assert self.grad is not None
        assert self.grad.shape == grad.shape

        self.grad += grad

        for dep in self.depends_on:
            local_grad = dep.gradfn(grad)
            dep.tensor.backward(local_grad)

        return

    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        data = self.data.reshape(shape)
        return Tensor(data, requires_grad=self.requires_grad)


    def __str__(self) -> str:
        return self.data.__str__()
        

    @staticmethod
    def random(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad)

    def zero_grad(self) -> None:
        assert self.requires_grad, "Cannot zero the gradients of a tensor that has requires_grad = False."

        self.grad = np.zeros_like(self.data, dtype=np.float64)
        return

    @property
    def T(self) -> 'Tensor':
        return Tensor(self.data.T)

    def __repr__(self) -> str:
        return f"<Tensor shape={self.shape} requires_grad={self.requires_grad}>"

    def __add__(self, other: Numeric) -> 'Tensor':
        from neuron import backend
        return backend.add(self, _ensure_tensor(other))
    
    def __neg__(self) -> 'Tensor':
        from neuron import backend
        return backend.neg(self)

    def __sub__(self, other: Numeric) -> 'Tensor':
        from neuron import backend
        return backend.sub(self, _ensure_tensor(other))

    def __radd__(self, other: Numeric) -> 'Tensor':
        from neuron import backend
        return backend.add(self, _ensure_tensor(other))

    def __mul__(self, other: Numeric) -> 'Tensor':
        from neuron import backend
        return backend.mul(self, _ensure_tensor(other))

    def __rmul__(self, other: Numeric) -> 'Tensor':
        from neuron import backend
        return backend.mul(self, _ensure_tensor(other))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        from neuron import backend
        return backend.matmul(self, other)

    def sum(self) -> 'Tensor':
        from neuron import backend
        return backend.sum(self)
        
    
def _ensure_array(data: Numeric) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, float | int):
        return np.array(data, dtype=np.float64)
    return np.array(data, dtype=np.float64)

def _ensure_tensor(data: Numeric) -> Tensor:
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)


