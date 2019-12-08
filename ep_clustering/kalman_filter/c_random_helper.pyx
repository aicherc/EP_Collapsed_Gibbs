# distutils: language = c++

# ^ distutils lines are important

# Import Eigen types (e.g. VectorXd, MatrixXd, numpy_copy, np.ndarray)
from eigency.core cimport *
from libcpp.vector cimport vector
import numpy as np
import logging


# Pythonic redefinition of Kalman Filter class header for Cython
cdef extern from "../cython/src/random_helper.h":
    cdef VectorXd _multivariate_normal_rv "multivariate_normal_rv" \
            (Map[VectorXd] &, Map[MatrixXd] &)
    cdef double  _normal_rv "normal_rv" \
            (double &, double &)

def c_multivariate_normal_rv(np.ndarray mean, np.ndarray variance):
    """ Eigen implementation of multivariate normal rv """

    if np.any(np.linalg.eigvals(variance) <= 0):
        raise ValueError("variance must be positive definite")
    if (np.size(mean) != np.shape(variance)[0] or
            np.size(mean) != np.shape(variance)[1]):
        raise ValueError("variance and mean don't match sizes")

    x = ndarray_copy(
            _multivariate_normal_rv(Map[VectorXd](mean),
                Map[MatrixXd](variance))
            ).T[0]
    return x

def c_normal_rv(mean, variance):
    """ Eigen implementation of normal rv """

    if(variance <= 0):
        raise ValueError("variance must be positive")

    return _normal_rv(<double> mean, <double> variance)



