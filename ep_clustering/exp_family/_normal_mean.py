#!/usr/bin/env python
"""
Natural Parameters Helper
"""

# Import Modules
import numpy as np
from ep_clustering._utils import fix_docs
from ._exponential_family import ExponentialFamily

# Author Information
__author__ = "Christopher Aicher"

# Normal Exponential Family
@fix_docs
class NormalFamily(ExponentialFamily):
    """ Normal Family Site Approximation Class

    Natural Parameters:
        'mean_precision': (num_dim ndarray)
        'precision': (num_dim by num_dim ndarray)
            (__init__ will cast input to matrix if diagonal matrix)

    Helper Functions:
        get_mean(): return mean
        get_variance(): return variance
    """
    def __init__(self, num_dim, log_scaling_coef=0.0, mean_precision=None,
            precision=None):
        # Set Default mean_precision and precision
        if mean_precision is None:
            mean_precision = np.zeros(num_dim)
        if precision is None:
            precision = np.zeros((num_dim, num_dim))

        # Check Dimensions
        if np.size(mean_precision) != num_dim:
            raise ValueError("mean_precision must be size num_dim")
        if np.shape(precision) != (num_dim, num_dim):
            if np.size(precision) == num_dim:
                precision = np.diag(precision)
            else:
                raise ValueError("precision must be size num_dim by num_dim")

        super(NormalFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                mean_precision=mean_precision,
                precision=precision
                )
        return

    def is_valid_density(self):
        eigvals = np.linalg.eigvals(self.natural_parameters['precision'])
        if np.any(eigvals < 1e-12):
            return False
        return True

    def sample(self):
        self._check_is_valid_density()
        L = np.linalg.cholesky(self.natural_parameters['precision'])
        z = np.random.normal(size = self.num_dim)
        sample = np.linalg.solve(L.T, z + np.linalg.solve(L.T,
            self.natural_parameters['mean_precision']))
        return sample

    def logpartition(self):
        self._check_is_valid_density()
        sign, logdet_precision = np.linalg.slogdet(
                self.natural_parameters['precision'])
        log_partition = 0.5 * self.num_dim * np.log(2 * np.pi)
        log_partition -= 0.5 * logdet_precision
        log_partition += 0.5 * self.natural_parameter['mean_precision'].dot(
                np.linalg.solve(self.natural_parameters['precision'],
                    self.natural_parameters['mean_precision']))
        return log_partition

    def get_mean(self):
        """ Return the mean """
        mean = self.natural_parameters['mean_precision'] / \
                self.natural_parameters['precision']
        return mean

    def get_variance(self):
        """ Return the variance """
        variance = self.natural_parameters['precision'] ** -1
        return variance

@fix_docs
class DiagNormalFamily(ExponentialFamily):
    """ Normal Family Site Approximation Class (with diag variance)

    Natural Parameters Args:
        'mean_precision' (num_dim ndarray): precision * mean
        'precision' (num_dim ndarray): diagonal precision

    Helper Functions:
        get_mean(): return mean
        get_variance(): return variance
    """
    def __init__(self, num_dim, log_scaling_coef=0.0, mean_precision=None,
            precision=None):
        # Set Default mean_precision and precision
        if mean_precision is None:
            mean_precision = np.zeros(num_dim)
        if precision is None:
            precision = np.zeros(num_dim)

        # Check Dimensions
        if np.size(mean_precision) != num_dim:
            raise ValueError("mean_precision must be size num_dim")
        if np.size(precision) != num_dim:
            raise ValueError("precision must be size num_dim")

        super(DiagNormalFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                mean_precision=mean_precision,
                precision=precision
                )
        return

    def is_valid_density(self):
        if np.any(self.natural_parameters['precision'] < 1e-12):
            return False
        return True

    def sample(self):
        self._check_is_valid_density()
        sample = self.get_mean() + \
                np.random.normal(size = self.num_dim) * \
                np.sqrt(self.get_variance())
        return sample

    def logpartition(self):
        self._check_is_valid_density()
        logdet_precision = np.sum(np.log(self.natural_parameters['precision']))
        log_partition = 0.5 * self.num_dim * np.log(2 * np.pi)
        log_partition -= 0.5 * logdet_precision
        log_partition += 0.5 * np.sum(
                self.natural_parameters['mean_precision'] ** 2 /
                self.natural_parameters['precision'])
        return log_partition

    def get_mean(self):
        """ Return the mean """
        mean = self.natural_parameters['mean_precision'] / \
                self.natural_parameters['precision']
        return mean

    def get_variance(self):
        """ Return the variance """
        variance = self.natural_parameters['precision'] ** -1
        return variance

    @staticmethod
    def from_mean_variance(mean, variance):
        """ Helper function for constructing DiagNormalFamily from
        mean and variance

        Args:
            mean (ndarray): mean (size num_dim)
            variance (ndarray): variance (size num_dim)
        """
        num_dim = np.size(mean)
        if np.size(variance) != num_dim:
            raise ValueError("mean and variance dimensions do not match")
        precision = 1.0/variance
        mean_precision = mean * precision
        return DiagNormalFamily(
                num_dim=num_dim,
                mean_precision=mean_precision,
                precision=precision,
                )







