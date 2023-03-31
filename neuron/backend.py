
from typing import Union, Callable
from neuron.tensor import Tensor, Dependency

import numpy as np

Numeric = Union[float, list[float], list[int], np.ndarray]
GradFn = Callable[[np.ndarray], np.ndarray]

def add(t1: Tensor, t2: Tensor) -> Tensor:

    requires_grad = t1.requires_grad or t2.requires_grad
    t = Tensor(t1.data + t2.data, requires_grad)

    if t1.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        t.depends_on.append(Dependency(t1, gradfn, "addBackward0"))

    if t2.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        t.depends_on.append(Dependency(t2, gradfn, "addBackward1"))

    return t

def mul(t1: Tensor, t2: Tensor) -> Tensor:
    requires_grad = t1.requires_grad or t2.requires_grad

    t = Tensor(t1.data * t2.data, requires_grad)

    if t1.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        t.depends_on.append(Dependency(t1, gradfn, "mulBackward0"))

    if t2.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        t.depends_on.append(Dependency(t2, gradfn, "mulBackward1"))
    return t

def neg(t0: Tensor) -> Tensor:
    t1 = Tensor(-t0.data, t0.requires_grad)

    if t0.requires_grad:
        gradfn: GradFn = lambda grad : -grad
        t1.depends_on.append(Dependency(t0, gradfn, "negBackward"))

    return t1

def sub(t0: Tensor, t1: Tensor) -> Tensor:
    t2 = Tensor(t0.data - t1.data, t0.requires_grad or t1.requires_grad)

    if t0.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t0.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t0.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        t2.depends_on.append(Dependency(t0, gradfn, "subBackward0"))

    if t1.requires_grad:
        def gradfn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return -grad
        t2.depends_on.append(Dependency(t1, gradfn, "subBackward1"))

    return t2

def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    assert t1.shape[1] == t2.shape[0], f"Cannot multiply tensor with shape {t1.shape} to {t2.shape}"

    requires_grad = t1.requires_grad or t2.requires_grad
    t = Tensor(t1.data @ t2.data, requires_grad)

    if t1.requires_grad:
        gradfn: GradFn = lambda grad : grad @ t2.data.T
        t.depends_on.append(Dependency(t1, gradfn, "matmulBackward0"))

    if t2.requires_grad:
        gradfn: GradFn = lambda grad : t1.data.T @ grad
        t.depends_on.append(Dependency(t2, gradfn, "matmulBackward1"))

    return t

def sum(t1: Tensor) -> Tensor:
    t = Tensor(t1.data.sum(), t1.requires_grad)

    if t1.requires_grad:
        gradfn: GradFn = lambda grad : grad * np.ones_like(t1.data)
        t.depends_on.append(Dependency(t1, gradfn, "sumBackward"))

    return t
