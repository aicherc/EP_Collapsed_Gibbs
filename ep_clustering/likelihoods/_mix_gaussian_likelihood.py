#!/usr/bin/env python
"""
Mixture of Scale Gaussians Likelihood
"""

# Import Modules
import numpy as np
import logging
from scipy.special import logsumexp
from ep_clustering._utils import fix_docs
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.exp_family._normal_mean import (
        NormalFamily,
        DiagNormalFamily
    )

logger = logging.getLogger(name=__name__)


@fix_docs
class MixDiagGaussianLikelihood(Likelihood):
    """ Mixture of Scale Gaussian Likelihood Object

    Args:
        **kwargs : parameters
            variance (double or ndarray): clutter noise, length num_dim
            proportions (ndarray): length num_scale_components
            scales (ndarray): length num_scale_components
                (!!Note!! that this scales the variance)
    """
    # Inherit Docstrings
    __doc__ += Likelihood.__doc__

    # Class Variables
    name = "MixScaleDiagGaussian"

    def __init__(self, data, **kwargs):
        self.y = data.matrix
        self.num_dim = data.num_dim
        super(MixDiagGaussianLikelihood, self).__init__(data, **kwargs)
        self.num_scale_components = np.size(self.parameter.proportions)
        return

    def _get_default_prior(self):
        theta_prior = DiagNormalFamily(num_dim = self.num_dim,
                precision=np.ones(self.num_dim)/100.0)
        return theta_prior

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        default_parameter = {
            "variance": np.ones(self.num_dim),
            "proportions": np.ones(2)/2.0,
            "scales": np.array([0.5, 1.5]),
            }
        return default_parameter

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        prior = {}
        return prior

    def _sample_from_prior(self):
        raise NotImplementedError(
                "Parameter prior not implemented for MixDiagGaussian")

    def loglikelihood(self, index, theta):
        y = self.y[index]
        component_loglikelihood = np.zeros(self.num_scale_components)
        for c in range(self.num_scale_components):
            variance = self.parameter.variance * self.parameter.scales[c]
            component_loglikelihood[c] = \
                    -0.5*((y-theta)/variance).dot(y-theta) + \
                    -0.5*self.num_dim*np.sqrt(2*np.pi) + \
                    -0.5*np.sum(np.log(variance))
            component_loglikelihood[c] += \
                    np.log(self.parameter.proportions[c])
        loglikelihood = logsumexp(component_loglikelihood)

        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        raise NotImplementedError("collapsed likelihood not implemented")

    def moment(self, index, theta_parameter):
        y = self.y[index]
        theta_parameter_log_partition = theta_parameter.logpartition()

        # Calculate Moments of Individual Components
        logZs = np.zeros(self.num_scale_components)
        means = np.zeros((self.num_scale_components, self.num_dim))
        second_moments = np.zeros((self.num_scale_components, self.num_dim))
        for c in range(self.num_scale_components):
            site_c = DiagNormalFamily.from_mean_variance(
                    mean=y,
                    variance=self.parameter.variance*self.parameter.scales[c],
                )
            post_approx_c = (theta_parameter + site_c)

            logZs[c] = post_approx_c.logpartition() - \
                    (theta_parameter_log_partition + site_c.logpartition())
            means[c] = post_approx_c.get_mean()
            second_moments[c] = \
                    post_approx_c.get_variance() + post_approx_c.get_mean()**2

        # Aggregate Moments
        weights = np.exp(logZs) * self.parameter.proportions
        Z = np.sum(weights)
        mean = np.sum(
                means * np.outer(weights, np.ones(self.num_dim)),
                axis=0,
                ) / Z
        second_moment = np.sum(
                second_moments * np.outer(weights, np.ones(self.num_dim)),
                axis=0,
                ) / Z
        variance = second_moment - mean**2

        # Construct and return the unnormalized_post_approx
        unnormalized_post_approx = DiagNormalFamily.from_mean_variance(
                mean = mean,
                variance = variance,
                )
        unnormalized_post_approx.log_scaling_coef = np.log(Z)
        return unnormalized_post_approx

    def sample(self, indices, prior_parameter):
        raise NotImplementedError("sample theta not implemented")

    def update_parameters(self, z, theta, parameter_name = None):
        raise NotImplementedError("update parameters not implemented")

# EOF
