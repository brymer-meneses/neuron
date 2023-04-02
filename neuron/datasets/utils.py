
import numpy as np
from neuron import Tensor

def encode_one_hot(num_classes: int, t: Tensor) -> Tensor:

    # https://stackoverflow.com/a/49790223
    one_hot = np.identity(num_classes)[t.data]

    return Tensor(one_hot)


