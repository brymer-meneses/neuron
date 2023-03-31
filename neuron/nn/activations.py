
from neuron.tensor import Dependency, Tensor
from neuron.backend import GradFn

import numpy as np

def ReLU(t0: Tensor) -> Tensor:

    data = np.maximum(0, t0.data, t0.data)
    t1 = Tensor(data, t0.requires_grad)

    if t0.requires_grad:
        gradfn: GradFn = lambda grad : grad * np.where(t0.data > 0, 1, 0)
        t1.depends_on.append(Dependency(t0, gradfn, "reluBackward"))

    return t1

def sigmoid(t0: Tensor) -> Tensor:

    _sigmoid = lambda x : 1 / (1 + np.exp(-x))
    
    data = 1 / (1 + np.exp(-t0.data))
    t1 = Tensor(data, t0.requires_grad)
        
    if t0.requires_grad:
        gradfn: GradFn = lambda grad : grad * _sigmoid(t0.data) * _sigmoid(1 - t0.data)
        t1.depends_on.append(Dependency(t0, gradfn, "sigmoidBackward"))
    return t1


def softmax(t0: Tensor) -> Tensor:

    # exp = np.exp(t0.data)
    # data = exp / exp.sum(axis=0)
    #
    # t1 = Tensor(data, t0.requires_grad)
    #
    # if t0.requires_grad:
    #     # TODO: implement gradient for softmax
    #     gradfn: GradFn = lambda grad : grad
    #     t1.depends_on.append(Dependency(t1, gradfn, "softmaxBackward"))
    #
    # return t1
    raise NotImplementedError
