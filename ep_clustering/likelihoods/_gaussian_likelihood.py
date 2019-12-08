#!/usr/bin/env python
"""
Gaussian Likelihood
"""

# Import Modules
import numpy as np
import logging
from ep_clustering._utils import fix_docs
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.exp_family._normal_mean import (
        NormalFamily,
        DiagNormalFamily
    )

logger = logging.getLogger(name=__name__)
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )



@fix_docs
class GaussianLikelihood(Likelihood):
    """ Gaussian Likelihood Object

    Args:
        **kwargs :
            variance (double) - cluster noise
    """
    # Inherit Docstrings
    __doc__ += Likelihood.__doc__

    # Class Variables
    name = "Gaussian"

    def __init__(self, data, **kwargs):
        self.y = data.matrix
        self.num_dim = data.num_dim
        super(GaussianLikelihood, self).__init__(data, **kwargs)
        return

    def _get_default_prior(self):
        theta_prior = DiagNormalFamily(num_dim = self.num_dim,
                precision=np.ones(self.num_dim)/100.0)
        return theta_prior

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        default_parameter = {
            "variance": np.ones(self.num_dim),
            }

        return default_parameter

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        prior = {
            "alpha_variance0": 3.0,
            "beta_variance0": 2.0,
            }
        return prior

    def _sample_from_prior(self):
        parameter = {
                    "variance": 1.0/np.random.gamma(
                        shape=self.prior.alpha_variance0,
                        scale=self.prior.beta_variance0,
                        size=self.num_dim)
                    }
        return parameter

    def loglikelihood(self, index, theta):
        y = self.y[index]
        loglikelihood = -0.5*((y-theta)/self.parameter.variance).dot(y-theta) +\
                -0.5*self.num_dim*np.log(2*np.pi) + \
                -0.5*np.sum(np.log(self.parameter.variance))

        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        posterior = theta_parameter
        for s_index in subset_indices:
            s_y = self.y[s_index]
            posterior = (posterior +
                    DiagNormalFamily.from_mean_variance(
                        mean=s_y,
                        variance=self.parameter.variance)
                    )
        mean = posterior.get_mean()
        variance = posterior.get_variance() + self.parameter.variance
        y = self.y[index]
        loglikelihood = -0.5*((y-mean)/variance).dot(y-mean) + \
                    -0.5*self.num_dim*np.log(2*np.pi) + \
                    -0.5*np.sum(np.log(variance))
        return loglikelihood

    def moment(self, index, theta_parameter):
        y = self.y[index]
        site = DiagNormalFamily.from_mean_variance(
                mean=y,
                variance=self.parameter.variance,
                )

        unnormalized_post_approx = (theta_parameter + site)
        unnormalized_post_approx.log_scaling_coef = \
                unnormalized_post_approx.logpartition() - \
                (theta_parameter.logpartition() + site.logpartition())
        return unnormalized_post_approx

    def sample(self, indices, prior_parameter):
        posterior = prior_parameter
        for index in indices:
            y = self.y[index]
            posterior = (posterior +
                    DiagNormalFamily.from_mean_variance(
                        mean=y,
                        variance=self.parameter.variance)
                    )
        mean = posterior.get_mean()
        variance = posterior.get_variance()
        return np.random.randn(posterior.num_dim)*np.sqrt(variance) + mean

    def update_parameters(self, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_variance(z, theta)
        elif parameter_name == "variance":
            self._update_variance(z, theta)
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return

    def _update_variance(self, z, theta, k_list=None):
        if k_list is None:
            k_list = range(np.shape(theta)[0])

        sse = 0
        N = 0
        for k in k_list:
            ind = (z == k)
            N += np.sum(ind)
            sse += np.sum((self.y[ind,:] - theta[k]) ** 2)

        alpha_variance = self.prior.alpha_variance0 + N/2.0
        beta_variance = self.prior.beta_variance0 + sse/2.0

        self.parameter.variance = 1.0/np.random.gamma(shape = alpha_variance,
                scale = 1.0/beta_variance, size = self.num_dim)
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_variance(z, theta, k_list[k])
        elif parameter_name == "variance":
            self._update_variance(z, theta, k_list[k])
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return



# EOF
