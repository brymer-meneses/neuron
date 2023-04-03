from unittest import TestCase

from neuron import Tensor
from neuron.nn.activations import ReLU, Sigmoid, TanH

import numpy as np

class TestActivations(TestCase):
    
    def test_relu(self):
        t0 = Tensor.random((3,3), requires_grad=True)
        t1 = ReLU(t0)

        t1.backward(np.ones((3,3)))

        np.testing.assert_equal(t1.data , np.where(t0.data > 0, t0.data, 0))

        assert t0.grad is not None
        assert t1.grad is not None

        np.testing.assert_equal(t0.grad, np.where(t0.data > 0, 1, 0))

    def test_tanh(self):
        t0 = Tensor.random((3,3), requires_grad=True)
        t1 = TanH(t0)

        t1.backward(np.ones((3,3)))

        np.testing.assert_equal(t1.data, np.tanh(t0.data))
        
        assert t0.grad is not None
        assert t1.grad is not None

        np.testing.assert_equal(t0.grad, 1 - np.tanh(t0.data) ** 2)

    def test_sigmoid(self):
        t0 = Tensor.random((3,3), requires_grad=True)
        t1 = Sigmoid(t0)

        t1.backward(np.ones((3,3)))

        _sigmoid = lambda x : 1 / (1 + np.exp(-x))

        np.testing.assert_equal(t1.data, _sigmoid(t0.data))
        
        assert t0.grad is not None
        assert t1.grad is not None

        np.testing.assert_equal(t0.grad, _sigmoid(t0.data) * _sigmoid(1-t0.data))

