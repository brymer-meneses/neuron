from neuron.tensor import Tensor, Dependency
from neuron.backend import GradFn
from typing import Callable

import numpy as np

Loss = Callable[[Tensor, Tensor], Tensor]

def MeanSquaredError(t0: Tensor, t1: Tensor) -> Tensor:

    assert t1.shape == t0.shape, "predicted and target tensors must have the same shape"

    data = np.square(t0.data - t1.data).mean()

    result = Tensor(data, t0.requires_grad or t1.requires_grad)
    num_elems = np.size(t0.data)

    if t0.requires_grad:
        gradfn: GradFn = lambda grad : grad * 2 * (1 / num_elems) * (t0.data - t1.data)
        result.depends_on.append(Dependency(t0, gradfn, "meanSquaredErrorBackward0"))

    if t1.requires_grad:
        gradfn: GradFn = lambda grad : grad * 2 * (1 / num_elems) * (t0.data - t1.data) * -1
        result.depends_on.append(Dependency(t1, gradfn, "meanSquaredErrorBackward1"))

    return result

