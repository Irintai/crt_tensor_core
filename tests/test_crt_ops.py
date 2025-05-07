# tests/test_crt_ops.py
import unittest
import math
from .. import Tensor, D, H, R, syntonic_stability
from .. import Dtype

class TestCRTOps(unittest.TestCase):
    def test_differentiation(self):
        # Real tensor
        t1 = Tensor([1.0, 2.0, 3.0])
        result = D(t1, alpha=0.5)
        
        # Check that result is complex
        self.assertTrue(result.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128])
        
        # Check values
        for i in range(len(t1.data)):
            expected = t1.data[i] * complex(1, 0.5)
            self.assertAlmostEqual(result.data[i].real, expected.real, places=5)
            self.assertAlmostEqual(result.data[i].imag, expected.imag, places=5)
        
        # Complex tensor
        t2 = Tensor([complex(1, 1), complex(2, 2)], dtype='complex64')
        result = D(t2, alpha=0.5)
        
        # Check values
        for i in range(len(t2.data)):
            expected = t2.data[i] * complex(1, 0.5)
            self.assertAlmostEqual(result.data[i].real, expected.real, places=5)
            self.assertAlmostEqual(result.data[i].imag, expected.imag, places=5)
    
    def test_harmonization(self):
        # Real tensor
        t1 = Tensor([1.0, 2.0, 3.0])
        result = H(t1, beta=0.7, gamma=0.3)
        
        # Check values
        for i in range(len(t1.data)):
            expected = t1.data[i] * complex(1, -0.7) + 0.3 * math.pi * t1.data[i]
            self.assertAlmostEqual(result.data[i].real, expected.real, places=5)
            self.assertAlmostEqual(result.data[i].imag, expected.imag, places=5)
        
        # Complex tensor
        t2 = Tensor([complex(1, 1), complex(2, 2)], dtype='complex64')
        result = H(t2, beta=0.7, gamma=0.3)
        
        # Check values
        for i in range(len(t2.data)):
            expected = t2.data[i] * complex(1, -0.7) + 0.3 * math.pi * t2.data[i]
            self.assertAlmostEqual(result.data[i].real, expected.real, places=5)
            self.assertAlmostEqual(result.data[i].imag, expected.imag, places=5)
    
    def test_recursion(self):
        # Real tensor
        t1 = Tensor([1.0, 2.0, 3.0])
        result = R(t1, alpha=0.5, beta=0.7, gamma=0.3)
        
        # Compare with manually applying D and H
        diff_result = D(t1, alpha=0.5)
        harm_result = H(diff_result, beta=0.7, gamma=0.3)
        
        # Check values
        for i in range(len(result.data)):
            self.assertAlmostEqual(result.data[i].real, harm_result.data[i].real, places=5)
            self.assertAlmostEqual(result.data[i].imag, harm_result.data[i].imag, places=5)
    
    def test_syntonic_stability(self):
        # Real tensor
        t1 = Tensor([1.0, 2.0, 3.0])
        stability = syntonic_stability(t1, alpha=0.5, beta=0.5, gamma=0.1)
        
        # Check that stability is in [0, 1]
        self.assertTrue(0 <= stability <= 1)
        
        # Test with identical D and H (perfect stability)
        t2 = Tensor([1.0])
        perfect_stability = syntonic_stability(t2, alpha=0, beta=0, gamma=0)
        self.assertAlmostEqual(perfect_stability, 1.0, places=5)
    
    def test_autograd(self):
        # Test autograd with CRT ops
        t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Forward pass
        diff_result = D(t1, alpha=0.5)
        harm_result = H(diff_result, beta=0.7, gamma=0.3)
        loss = (harm_result.sum() - Tensor(10.0)).sum()  # scalar
        
        # Backward pass
        loss.backward()
        
        # Check that t1 has gradients
        self.assertTrue(t1.grad is not None)

if __name__ == '__main__':
    unittest.main()