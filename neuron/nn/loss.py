from neuron.tensor import Tensor, Dependency
from neuron.backend import GradFn
from typing import Callable

import numpy as np

Loss = Callable[[Tensor, Tensor], Tensor]

def MeanSquaredError(predicted: Tensor, target: Tensor) -> Tensor:

    assert predicted.shape == target.shape, "predicted and target tensors must have the same shape"

    data = np.power(predicted.data - target.data, 2).mean()
    result = Tensor(data, predicted.requires_grad)

    if predicted.requires_grad:
        gradfn: GradFn = lambda grad : grad * 2 * (1 / np.size(predicted.data))
        result.depends_on.append(Dependency(predicted, gradfn, "meanSquaredErrorBackward"))

    return result

