from unittest import TestCase

from neuron import Tensor
import numpy as np

class TensorOpsTest(TestCase): 
    def test_simple_tensor_sum(self):
        t0 = Tensor.random((1, 5), requires_grad=True)

        t1 = t0.sum()
        t1.backward()

        np.testing.assert_array_equal(t1.data, t0.data.sum())
        assert t0.grad is not None
        assert t1.grad is not None
        np.testing.assert_array_equal(t0.grad, np.ones((1, 5)))

    def test_tensor_sum_with_grad(self):
        t0 = Tensor.random((1, 5), requires_grad=True)

        t1 = t0.sum()
        t1.backward(3)


        assert t0.grad is not None
        assert t1.grad is not None

        np.testing.assert_array_equal(t1.data, t0.data.sum())
        np.testing.assert_array_equal(t0.grad, 3*np.ones((1, 5)))

    def test_tensor_simple_add(self):
        t0 = Tensor.random((3, 3), requires_grad=True)
        t1 = Tensor.random((3, 3), requires_grad=True)
        t2 = t0 + t1

        initial_grad = np.ones((3,3))
        t2.backward(initial_grad)

        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t2.data, t0.data + t1.data)
        np.testing.assert_array_equal(t0.grad, initial_grad)
        np.testing.assert_array_equal(t1.grad, initial_grad)

    def test_tensor_broadcasted_add1(self):
        t0 = Tensor(1, requires_grad=True)
        t1 = Tensor.random((3,3), requires_grad=True)

        t2 = t0 + t1
        initial_grad = np.ones((3,3))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data + t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, np.ones_like(t1.data).sum(axis=(0,1)))
        np.testing.assert_array_equal(t1.grad, initial_grad)

    def test_tensor_broadcasted_add2(self):
        t0 = Tensor.random((1 ,3), requires_grad=True)
        t1 = Tensor.random((10, 3), requires_grad=True)

        t2 = t0 + t1
        initial_grad = np.ones((10,3))
        t2.backward(initial_grad)

        assert np.array_equal(t2.data, t0.data + t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, 10*np.ones_like(t0.data))
        np.testing.assert_array_equal(t1.grad, np.ones_like(t1.data))

    def test_tensor_neg(self):
        t0 = Tensor.random((3 ,3), requires_grad=True)
        t1 = -t0

        initial_grad = np.ones((3,3))
        t1.backward(initial_grad)

        np.testing.assert_array_equal(t1.data, -t0.data)
        assert t0.grad is not None
        assert t1.grad is not None

        np.testing.assert_array_equal(t0.grad, -initial_grad)
        np.testing.assert_array_equal(t1.grad, initial_grad)

    def test_tensor_simple_sub(self):
        t0 = Tensor.random((3,3), requires_grad=True)
        t1 = Tensor.random((3,3), requires_grad=True)
        t2 = t0 - t1

        initial_grad = np.ones((3,3))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data - t1.data)

        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, initial_grad)
        np.testing.assert_array_equal(t1.grad, -initial_grad)

    def test_tensor_broadcasted_sub1(self):
        t0 = Tensor(1, requires_grad=True)
        t1 = Tensor.random((3,3), requires_grad=True)

        t2 = t0 - t1

        initial_grad = np.ones((3,3))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data - t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, np.ones_like(t1.data).sum(axis=(0,1)))
        np.testing.assert_array_equal(t1.grad, -initial_grad)

    def test_tensor_broadcasted_sub2(self):
        t0 = Tensor.random((1 ,3), requires_grad=True)
        t1 = Tensor.random((10, 3), requires_grad=True)

        t2 = t0 - t1
        initial_grad = np.ones((10,3))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data - t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, 10*np.ones_like(t0.data))
        np.testing.assert_array_equal(t1.grad, -np.ones_like(t1.data))

    def test_tensor_simple_mul(self):

        t0 = Tensor.random((3,3), requires_grad=True)
        t1 = Tensor.random((3,3), requires_grad=True)
        t2 = t0 * t1

        initial_grad = np.ones((3,3))
        t2.backward(initial_grad)

        assert np.array_equal(t2.data, t0.data * t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        print(t0.data)
        print(t1.data)

        np.testing.assert_array_equal(t0.grad, initial_grad * t1.data)
        np.testing.assert_array_equal(t1.grad, initial_grad * t0.data)

    def test_tensor_broadcasted_mul1(self):
        t0 = Tensor.random((1 ,3), requires_grad=True)
        t1 = Tensor.random((10, 3), requires_grad=True)

        t2 = t0 * t1
        initial_grad = np.ones((10,3))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data * t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        t0_grad = np.expand_dims(np.sum(initial_grad * t1.data, axis=0), axis=0)

        np.testing.assert_array_equal(t1.grad, initial_grad * t0.data)
        np.testing.assert_array_equal(t0.grad, t0_grad)

    def test_tensor_broadcasted_mul2(self):

        t0 = Tensor(5, requires_grad=True)
        t1 = Tensor.random((10,10), requires_grad=True)

        t2 = t0 * t1
        initial_grad = np.ones((10,10))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data * t1.data)

        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_almost_equal(t1.grad, initial_grad * t0.data)
        np.testing.assert_array_almost_equal(t0.grad, np.sum(t1.data, axis=(0,1)))



    def test_tensor_matmul(self):
        t0 = Tensor.random((2, 5), requires_grad=True)
        t1 = Tensor.random((5, 1), requires_grad=True)
        t2 = t0 @ t1

        initial_grad = np.ones((2, 1))
        t2.backward(initial_grad)

        np.testing.assert_array_equal(t2.data, t0.data @ t1.data)
        assert t0.grad is not None
        assert t1.grad is not None
        assert t2.grad is not None

        np.testing.assert_array_equal(t0.grad, initial_grad @ t1.data.T)
        np.testing.assert_array_equal(t1.grad, t0.data.T @ initial_grad)

