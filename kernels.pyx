# kernels.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef differentiation_kernel(double[:] tensor_data, double[:] result_data, double alpha, int size):
    """
    Cython implementation of the differentiation kernel.
    
    Args:
        tensor_data: Flat tensor data array
        result_data: Pre-allocated output array
        alpha: Differentiation strength coefficient
        size: Size of the data arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = tensor_data[i] * (1.0 + alpha * 1j)
        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef harmonization_kernel(double[:] tensor_data, double[:] result_data, double beta, double gamma, int size):
    """
    Cython implementation of the harmonization kernel.
    
    Args:
        tensor_data: Flat tensor data array
        result_data: Pre-allocated output array
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        size: Size of the data arrays
    """
    cdef int i
    cdef double PI = 3.14159265358979323846
    for i in range(size):
        result_data[i] = tensor_data[i] * (1.0 - beta * 1j) + gamma * PI * tensor_data[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef recursion_kernel(double[:] tensor_data, double[:] result_data, double alpha, double beta, double gamma, int size):
    """
    Cython implementation of the recursion kernel.
    
    Args:
        tensor_data: Flat tensor data array
        result_data: Pre-allocated output array
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        size: Size of the data arrays
    """
    cdef int i
    cdef double PI = 3.14159265358979323846
    cdef double complex temp
    for i in range(size):
        # Differentiation step
        temp = tensor_data[i] * (1.0 + alpha * 1j)
        # Harmonization step
        result_data[i] = temp * (1.0 - beta * 1j) + gamma * PI * temp