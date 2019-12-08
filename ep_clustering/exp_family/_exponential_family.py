#!/usr/bin/env python
"""
Exponential Family Class
"""

# Import Modules
import numpy as np
from ep_clustering._utils import fix_docs
from copy import deepcopy

# Author Information
__author__ = "Christopher Aicher"

class ExponentialFamily(object):
    """ Exponential Family Site Approximation Class

    Let X be a r.v.
    Pr(X) \propto \exp(natural_parameters*sufficient statistics - logpartition)

    Args:
        num_dim (int): dimension of random variable
        **kwargs: additional key-word args
            initialization of a natural_parameters by key name

    Attributes:
        natural_parameters (dict): natural parameters
        log_scaling_coef (double): scaling coeffient for site approximations

    Methods:
        copy()
        sample()
        logpartition()
        is_valid_density()
        normalize()
        __add__ and __sub__
    """
    def __init__(self, num_dim, log_scaling_coef=0.0,
            **kwargs):
        self.num_dim = num_dim
        self.log_scaling_coef = log_scaling_coef
        self.natural_parameters = dict(**kwargs)
        self._is_known_valid_density = False
        return

    def copy(self):
        """ Return a copy of the object """
        cls = type(self)
        my_copy = cls(num_dim = self.num_dim,
                log_scaling_coef = self.log_scaling_coef,
                **deepcopy(self.natural_parameters)
                )
        return my_copy

    def is_valid_density(self):
        """ Check if the object is a valid probability density """
        raise NotImplementedError()

    def _check_is_valid_density(self):
        if self._is_known_valid_density:
            return
        else:
            self._is_known_valid_density = self.is_valid_density()
            if not self._is_known_valid_density:
                raise RuntimeError("ExponentialFamily is not a valid density")
        return

    def sample(self):
        """ Return a sample draw from the ExponentialFamily """
        raise NotImplementedError()

    def logpartition(self):
        """ Evaluates the logpartition function for current parameters """
        raise NotImplementedError()

    def normalize(self):
        """ Set log_scaling_coef so the ExponentialFamily integrates to 1

        Must be a valid density

        """
        self._check_is_valid_density()
        self.log_scaling_coef = 0.0
        return self

    def __add__(self, right):
        result = self.copy()
        result += right
        return result

    def __iadd__(self, right):
        cls = type(self)
        if not isinstance(right, cls):
            raise TypeError("RHS must be type {0}".format(cls))
        if self.num_dim != right.num_dim:
            raise ValueError(
                "Dimensions of LHS and RHS do not match: {0} != {1}".format(
                    self.num_dim, right.num_dim)
                )

        for key in self.natural_parameters.keys():
            self.natural_parameters[key] += right.natural_parameters[key]

        self.log_scaling_coef += right.log_scaling_coef
        self._is_known_valid_density = False
        return self

    def __sub__(self, right):
        result = self.copy()
        result -= right
        return result

    def __isub__(self, right):
        cls = type(self)
        if not isinstance(right, cls):
            raise TypeError("RHS must be type {0}".format(cls))
        if self.num_dim != right.num_dim:
            raise ValueError(
                "Dimensions of LHS and RHS do not match: {0} != {1}".format(
                    self.num_dim, right.num_dim)
                )

        for key in self.natural_parameters.keys():
            self.natural_parameters[key] -= right.natural_parameters[key]

        self.log_scaling_coef -= right.log_scaling_coef
        self._is_known_valid_density = False
        return self

    def __mul__(self, right):
        result = self.copy()
        result *= right
        return result

    def __rmul__(self, left):
        result = self.copy()
        result *= left
        return result

    def __imul__(self, right):
        if not isinstance(right, float):
            raise TypeError(
                "Multiplication only defined for floats not {0}".format(right)
                )
        for key in self.natural_parameters.keys():
            self.natural_parameters[key] *= right
        self.log_scaling_coef *= right
        self._is_known_valid_density = False
        return self

    def __repr__(self):
        obj_repr = super(ExponentialFamily, self).__repr__()
        obj_repr += "\nnum_dim: " + str(self.num_dim)
        obj_repr += "\nlog_scale_coef: " + str(self.log_scaling_coef)
        for para_name, para_value in self.natural_parameters.items():
            obj_repr += "\n{0}:\n{1}".format(para_name, para_value)
        obj_repr += "\n"
        return obj_repr

    def as_vector(self):
        """ Return vector of log_scaling_coef and natural_parameters """
        vector_list = [np.array([self.log_scaling_coef])]
        for key, value in self.natural_parameters.items():
            vector_list.append(value.flatten())
        vector = np.concatenate(vector_list)
        return vector


