from unittest import TestCase

from neuron import Tensor
from neuron.nn.activations import ReLU, Sigmoid, TanH
from neuron.nn.loss import MeanSquaredError

import numpy as np


class TestLoss(TestCase):
    def test_mean_squared_error(self):

        data0 = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]

        data1 = [[10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]

        t0 = Tensor(data0, requires_grad=True)
        t1 = Tensor(data1, requires_grad=True)


        t2 = MeanSquaredError(t0, t1)
        t2.backward()
        
        np.testing.assert_equal(t2.data, 81)

        np.testing.assert_equal(t1.grad, -(2/ np.size(t0.data)) * (t0.data - t1.data))
        np.testing.assert_equal(t0.grad, (2/ np.size(t0.data)) * (t0.data - t1.data))
