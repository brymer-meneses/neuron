
from typing import Iterator, NamedTuple
from neuron.tensor import Tensor

import numpy as np

class Batch(NamedTuple):
    inputs: Tensor
    targets: Tensor

class BatchIterator:

    def __init__(self, batch_size: int, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> Iterator[Batch]:
        assert len(inputs) == len(targets), "inputs and targets must have the same length"

        starting_points = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starting_points)

        for start in starting_points:
            end = start + self.batch_size
            
            batch_inputs = Tensor(inputs[start:end])
            batch_targets = Tensor(targets[start:end])

            yield Batch(batch_inputs, batch_targets)

