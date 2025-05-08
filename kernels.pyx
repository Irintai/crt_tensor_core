"""
Optimized Cython implementations of core CRT operations.

These kernels provide high-performance CPU implementations of the most
computationally intensive operations in the CRT library.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, log, exp, cos, sin, fabs, M_PI

cdef double PI = 3.14159265358979323846


#############################
# Basic Tensor Operations
#############################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void element_wise_add(double[:] a_data, double[:] b_data, double[:] result_data, int size) nogil:
    """
    Element-wise addition of two tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] + b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void element_wise_subtract(double[:] a_data, double[:] b_data, double[:] result_data, int size) nogil:
    """
    Element-wise subtraction of two tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] - b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void element_wise_multiply(double[:] a_data, double[:] b_data, double[:] result_data, int size) nogil:
    """
    Element-wise multiplication of two tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] * b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void element_wise_divide(double[:] a_data, double[:] b_data, double[:] result_data, int size) nogil:
    """
    Element-wise division of two tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] / b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void matrix_multiply(double[:] a_data, double[:] b_data, double[:] result_data, 
                         int a_rows, int a_cols, int b_cols) nogil:
    """
    Matrix multiplication of two 2D tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        a_rows: Number of rows in first tensor
        a_cols: Number of columns in first tensor
        b_cols: Number of columns in second tensor
    """
    cdef int i, j, k
    cdef double sum_val
    
    for i in range(a_rows):
        for j in range(b_cols):
            sum_val = 0.0
            for k in range(a_cols):
                sum_val += a_data[i * a_cols + k] * b_data[k * b_cols + j]
            result_data[i * b_cols + j] = sum_val


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double tensor_sum(double[:] data, int size) nogil:
    """
    Compute the sum of a tensor.
    
    Args:
        data: Tensor data (flat)
        size: Size of the array
        
    Returns:
        Sum of all elements
    """
    cdef int i
    cdef double sum_val = 0.0
    
    for i in range(size):
        sum_val += data[i]
    
    return sum_val


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double tensor_mean(double[:] data, int size) nogil:
    """
    Compute the mean of a tensor.
    
    Args:
        data: Tensor data (flat)
        size: Size of the array
        
    Returns:
        Mean of all elements
    """
    if size == 0:
        return 0.0
    return tensor_sum(data, size) / size


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double tensor_norm(double[:] data, int size) nogil:
    """
    Compute the L2 norm of a tensor.
    
    Args:
        data: Tensor data (flat)
        size: Size of the array
        
    Returns:
        L2 norm
    """
    cdef int i
    cdef double sum_sq = 0.0
    
    for i in range(size):
        sum_sq += data[i] * data[i]
    
    return sqrt(sum_sq)


#############################
# Complex Tensor Operations
#############################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void complex_add(double complex[:] a_data, double complex[:] b_data, 
                     double complex[:] result_data, int size) nogil:
    """
    Element-wise addition of two complex tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] + b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void complex_subtract(double complex[:] a_data, double complex[:] b_data, 
                          double complex[:] result_data, int size) nogil:
    """
    Element-wise subtraction of two complex tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] - b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void complex_multiply(double complex[:] a_data, double complex[:] b_data, 
                          double complex[:] result_data, int size) nogil:
    """
    Element-wise multiplication of two complex tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] * b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void complex_divide(double complex[:] a_data, double complex[:] b_data, 
                        double complex[:] result_data, int size) nogil:
    """
    Element-wise division of two complex tensors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        result_data: Pre-allocated output array
        size: Size of the arrays
    """
    cdef int i
    for i in range(size):
        result_data[i] = a_data[i] / b_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex tensor_complex_dot(double complex[:] a_data, double complex[:] b_data, int size) nogil:
    """
    Complex dot product of two vectors.
    
    Args:
        a_data: First tensor data (flat)
        b_data: Second tensor data (flat)
        size: Size of the arrays
        
    Returns:
        Complex dot product
    """
    cdef int i
    cdef double complex result = 0.0 + 0.0j
    
    for i in range(size):
        result += a_data[i] * b_data[i].conjugate()
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex_tensor_norm(double complex[:] data, int size) nogil:
    """
    Compute the L2 norm of a complex tensor.
    
    Args:
        data: Tensor data (flat)
        size: Size of the array
        
    Returns:
        L2 norm (real value)
    """
    cdef int i
    cdef double sum_sq = 0.0
    cdef double complex val
    
    for i in range(size):
        val = data[i]
        sum_sq += (val.real * val.real + val.imag * val.imag)
    
    return sqrt(sum_sq)


#############################
# CRT Specific Operations
#############################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void differentiation_kernel(double complex[:] tensor_data, double complex[:] result_data, 
                                double alpha, int size) nogil:
    """
    Cython implementation of the differentiation kernel.
    
    Args:
        tensor_data: Input tensor data (flat)
        result_data: Pre-allocated output array
        alpha: Differentiation strength coefficient
        size: Size of the arrays
    """
    cdef int i
    cdef double complex scale = 1.0 + alpha * 1.0j
    
    for i in range(size):
        result_data[i] = tensor_data[i] * scale


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void harmonization_kernel(double complex[:] tensor_data, double complex[:] result_data, 
                              double beta, double gamma, int size) nogil:
    """
    Cython implementation of the harmonization kernel.
    
    Args:
        tensor_data: Input tensor data (flat)
        result_data: Pre-allocated output array
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        size: Size of the arrays
    """
    cdef int i
    cdef double complex scale = 1.0 - beta * 1.0j
    
    for i in range(size):
        result_data[i] = tensor_data[i] * scale + gamma * PI * tensor_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void recursion_kernel(double complex[:] tensor_data, double complex[:] result_data, 
                          double alpha, double beta, double gamma, int size) nogil:
    """
    Cython implementation of the recursion kernel.
    
    Args:
        tensor_data: Input tensor data (flat)
        result_data: Pre-allocated output array
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        size: Size of the arrays
    """
    cdef int i
    cdef double complex diff_temp
    
    for i in range(size):
        # Differentiation step
        diff_temp = tensor_data[i] * (1.0 + alpha * 1.0j)
        
        # Harmonization step
        result_data[i] = diff_temp * (1.0 - beta * 1.0j) + gamma * PI * diff_temp


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void scale_dependent_differentiation_kernel(double complex[:] tensor_data, 
                                               double complex[:] result_data,
                                               double alpha, double scale,
                                               int size) nogil:
    """
    Cython implementation of the scale-dependent differentiation kernel.
    
    Args:
        tensor_data: Input tensor data (flat)
        result_data: Pre-allocated output array
        alpha: Differentiation strength coefficient
        scale: Scale parameter σ
        size: Size of the arrays
    """
    cdef int i
    cdef double complex coef = 1.0 + (alpha / scale) * 1.0j
    
    for i in range(size):
        result_data[i] = tensor_data[i] * coef


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void scale_dependent_harmonization_kernel(double complex[:] tensor_data, 
                                             double complex[:] result_data,
                                             double beta, double gamma, double scale,
                                             int size) nogil:
    """
    Cython implementation of the scale-dependent harmonization kernel.
    
    Args:
        tensor_data: Input tensor data (flat)
        result_data: Pre-allocated output array
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        scale: Scale parameter σ
        size: Size of the arrays
    """
    cdef int i
    cdef double complex h_coef = 1.0 - (beta * scale) * 1.0j
    cdef double syntony_coef = gamma * scale * PI
    
    for i in range(size):
        result_data[i] = tensor_data[i] * h_coef + syntony_coef * tensor_data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double calculate_syntonic_stability(double complex[:] diff_data, double complex[:] harm_data, 
                                        int size) nogil:
    """
    Calculate the syntonic stability index.
    
    Args:
        diff_data: Differentiated tensor data
        harm_data: Harmonized tensor data
        size: Size of the arrays
        
    Returns:
        Syntonic stability index in range [0, 1]
    """
    cdef double diff_norm = 0.0
    cdef double diff_harm_norm = 0.0
    cdef double complex diff_val, harm_val, diff_harm_diff
    cdef int i
    
    for i in range(size):
        diff_val = diff_data[i]
        diff_norm += (diff_val.real * diff_val.real + diff_val.imag * diff_val.imag)
        
        harm_val = harm_data[i]
        diff_harm_diff = diff_val - harm_val
        diff_harm_norm += (diff_harm_diff.real * diff_harm_diff.real + 
                           diff_harm_diff.imag * diff_harm_diff.imag)
    
    diff_norm = sqrt(diff_norm)
    diff_harm_norm = sqrt(diff_harm_norm)
    
    if diff_norm < 1e-10:
        return 1.0
    
    return 1.0 - (diff_harm_norm / diff_norm)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double phase_cycle_difference(double complex[:] tensor_data, double complex[:] result_data, 
                                  int size) nogil:
    """
    Calculate the phase-cycle difference for the i≈π relationship.
    
    Args:
        tensor_data: Input tensor data
        result_data: Pre-allocated array for temporary results
        size: Size of the arrays
        
    Returns:
        Norm of the difference ||P²[ψ] - C[ψ]||
    """
    cdef int i
    cdef double complex p2_val, c_val
    cdef double sum_sq_diff = 0.0
    
    for i in range(size):
        # P²[ψ] = (i·i)·ψ = -ψ
        p2_val = -tensor_data[i]
        
        # C[ψ] = e^(i·π)·ψ = -ψ
        c_val = -tensor_data[i]
        
        # Store the difference in result_data
        result_data[i] = p2_val - c_val
        
        # Accumulate squared difference
        sum_sq_diff += (result_data[i].real * result_data[i].real + 
                         result_data[i].imag * result_data[i].imag)
    
    return sqrt(sum_sq_diff)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double box_count_1d(double[:] tensor_data, int size, int box_size) nogil:
    """
    Count occupied boxes in a 1D array for fractal dimension calculation.
    
    Args:
        tensor_data: Input tensor data
        size: Size of the array
        box_size: Size of each box
        
    Returns:
        Number of occupied boxes
    """
    cdef int i, j
    cdef int count = 0
    cdef double threshold = 1e-10
    cdef bint has_value
    
    for i in range(0, size, box_size):
        has_value = False
        for j in range(i, min(i + box_size, size)):
            if fabs(tensor_data[j]) > threshold:
                has_value = True
                break
        if has_value:
            count += 1
    
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double box_count_2d(double[:, :] tensor_data, int height, int width, int box_size) nogil:
    """
    Count occupied boxes in a 2D array for fractal dimension calculation.
    
    Args:
        tensor_data: Input 2D tensor data
        height: Height of the array
        width: Width of the array
        box_size: Size of each box
        
    Returns:
        Number of occupied boxes
    """
    cdef int i, j, k, l
    cdef int count = 0
    cdef double threshold = 1e-10
    cdef bint has_value
    
    for i in range(0, height, box_size):
        for j in range(0, width, box_size):
            has_value = False
            for k in range(i, min(i + box_size, height)):
                for l in range(j, min(j + box_size, width)):
                    if fabs(tensor_data[k, l]) > threshold:
                        has_value = True
                        break
                if has_value:
                    break
            if has_value:
                count += 1
    
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void alpha_profile_kernel(double[:] syntony_values, double[:] result_values, 
                              double alpha_0, double gamma_alpha, int size) nogil:
    """
    Compute α(S) = α₀·(1-S)^γₐ coefficients for array of syntony values.
    
    Args:
        syntony_values: Array of syntonic indices
        result_values: Pre-allocated output array
        alpha_0: Base alpha coefficient
        gamma_alpha: Alpha scaling exponent
        size: Size of the arrays
    """
    cdef int i
    
    for i in range(size):
        result_values[i] = alpha_0 * pow(1.0 - syntony_values[i], gamma_alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void beta_profile_kernel(double[:] syntony_values, double[:] result_values, 
                             double beta_0, double kappa, int size) nogil:
    """
    Compute β(S) = β₀·(1-e^(-κ·S)) coefficients for array of syntony values.
    
    Args:
        syntony_values: Array of syntonic indices
        result_values: Pre-allocated output array
        beta_0: Base beta coefficient
        kappa: Exponential scaling factor
        size: Size of the arrays
    """
    cdef int i
    
    for i in range(size):
        result_values[i] = beta_0 * (1.0 - exp(-kappa * syntony_values[i]))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void gamma_profile_kernel(double[:] syntony_values, double[:] result_values, 
                              double gamma_0, double lambda_val, int size) nogil:
    """
    Compute γ(S) = γ₀·tanh(λ·S) coefficients for array of syntony values.
    
    Args:
        syntony_values: Array of syntonic indices
        result_values: Pre-allocated output array
        gamma_0: Base gamma coefficient
        lambda_val: Hyperbolic tangent scaling factor
        size: Size of the arrays
    """
    cdef int i
    cdef double s, tanh_val
    
    for i in range(size):
        s = syntony_values[i]
        # Calculate tanh manually to avoid dependency
        tanh_val = (exp(lambda_val * s) - exp(-lambda_val * s)) / (exp(lambda_val * s) + exp(-lambda_val * s))
        result_values[i] = gamma_0 * tanh_val