import unittest
import math
from .. import Tensor, D, H, R, syntonic_stability

class TestTensor(unittest.TestCase):
    def test_creation(self):
        # Test scalar creation
        t1 = Tensor(5.0)
        self.assertEqual(t1.shape, ())
        self.assertEqual(t1.data[0], 5.0)
        
        # Test vector creation
        t2 = Tensor([1.0, 2.0, 3.0])
        self.assertEqual(t2.shape, (3,))
        self.assertEqual(t2.data, [1.0, 2.0, 3.0])
        
        # Test matrix creation
        t3 = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t3.shape, (2, 2))
        self.assertEqual(t3.data, [1.0, 2.0, 3.0, 4.0])
    
    def test_zeros_ones(self):
        # Test zeros
        t1 = Tensor.zeros((2, 3))
        self.assertEqual(t1.shape, (2, 3))
        self.assertEqual(t1.data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Test ones
        t2 = Tensor.ones((2, 2))
        self.assertEqual(t2.shape, (2, 2))
        self.assertEqual(t2.data, [1.0, 1.0, 1.0, 1.0])
    
    def test_arithmetic(self):
        # Test addition
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([3.0, 4.0])
        t3 = t1 + t2
        self.assertEqual(t3.shape, (2,))
        self.assertEqual(t3.data, [4.0, 6.0])
        
        # Test subtraction
        t4 = t2 - t1
        self.assertEqual(t4.shape, (2,))
        self.assertEqual(t4.data, [2.0, 2.0])
        
        # Test multiplication
        t5 = t1 * t2
        self.assertEqual(t5.shape, (2,))
        self.assertEqual(t5.data, [3.0, 8.0])
        
        # Test division
        t6 = t2 / t1
        self.assertEqual(t6.shape, (2,))
        self.assertEqual(t6.data, [3.0, 2.0])
    
    def test_matmul(self):
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = Tensor([[5.0, 6.0], [7.0, 8.0]])
        t3 = t1 @ t2
        self.assertEqual(t3.shape, (2, 2))
        self.assertEqual(t3.data, [19.0, 22.0, 43.0, 50.0])
    
    def test_reshape(self):
        t1 = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        t2 = t1.reshape(2, 3)
        self.assertEqual(t2.shape, (2, 3))
        self.assertEqual(t2.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    def test_transpose(self):
        t1 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t2 = t1.transpose(0, 1)
        self.assertEqual(t2.shape, (3, 2))
        self.assertEqual(t2.data, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    
    def test_autograd(self):
        # Test basic gradient
        x = Tensor(2.0, requires_grad=True)
        y = x * x * x
        y.backward()
        self.assertEqual(x.grad.data[0], 12.0)  # Derivative of x^3 at x=2 is 3*2^2 = 12
        
        # Test more complex gradient
        a = Tensor(3.0, requires_grad=True)
        b = Tensor(4.0, requires_grad=True)
        c = a * a + b * b
        c.backward()
        self.assertEqual(a.grad.data[0], 6.0)  # Derivative of a^2 at a=3 is 2*3 = 6
        self.assertEqual(b.grad.data[0], 8.0)  # Derivative of b^2 at b=4 is 2*4 = 8

class TestCRTOperations(unittest.TestCase):
    def test_differentiation(self):
        # Test D operator
        psi = Tensor(1.0, dtype="complex64")
        alpha = Tensor(0.5)
        result = D(psi, alpha)
        
        # Expected: (1 + 0.5i) * 1.0 = 1.0 + 0.5i
        self.assertEqual(result.data[0], complex(1.0, 0.5))
    
    def test_harmonization(self):
        # Test H operator
        psi = Tensor(1.0, dtype="complex64")
        beta = Tensor(0.3)
        gamma = Tensor(0.2)
        result = H(psi, beta, gamma)
        
        # Expected: (1 - 0.3i) * 1.0 + 0.2 * π * 1.0 = 1.0 - 0.3i + 0.2π
        expected = complex(1.0 + 0.2 * math.pi, -0.3)
        self.assertAlmostEqual(result.data[0].real, expected.real)
        self.assertAlmostEqual(result.data[0].imag, expected.imag)
    
    def test_recursion(self):
        # Test R operator
        psi = Tensor(1.0, dtype="complex64")
        alpha = Tensor(0.5)
        beta = Tensor(0.3)
        gamma = Tensor(0.2)
        
        # Apply D first
        diff_result = D(psi, alpha)  # Expected: 1.0 + 0.5i
        
        # Apply H to the result of D
        expected_result = H(diff_result, beta, gamma)
        
        # Apply R directly
        actual_result = R(psi, alpha, beta, gamma)
        
        # Compare results
        self.assertAlmostEqual(actual_result.data[0].real, expected_result.data[0].real)
        self.assertAlmostEqual(actual_result.data[0].imag, expected_result.data[0].imag)
    
    def test_autograd_with_crt(self):
        # Test autograd with CRT operations
        psi = Tensor(1.0, dtype="complex64", requires_grad=True)
        alpha = Tensor(0.5, requires_grad=True)
        beta = Tensor(0.3, requires_grad=True)
        gamma = Tensor(0.2, requires_grad=True)
        
        # Apply R
        result = R(psi, alpha, beta, gamma)
        
        # Compute loss (for testing, use the real part)
        loss = Tensor(result.data[0].real)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(psi.grad)
        self.assertIsNotNone(alpha.grad)
        self.assertIsNotNone(beta.grad)
        self.assertIsNotNone(gamma.grad)

if __name__ == '__main__':
    unittest.main()