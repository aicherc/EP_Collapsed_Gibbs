#!/usr/bin/env python
"""
AR Likelihood
"""

# Import Modules
import numpy as np
import logging
from ep_clustering._utils import fix_docs
#from ep_clustering._natural_parameters import NaturalParameters
from ep_clustering.likelihoods._likelihoods import Likelihood

logger = logging.getLogger(name=__name__)

@fix_docs
class ARLikelihood(Likelihood):
    """ AR Likelihood

    Model: y_t = a y_{t-1} + lambduh theta_t + \epsilon_x

    Note: the first element of `theta`, `theta[k,0]`, should be ignored as it
    has no meaning for this model

    Args:
        **kwargs :
            - A (N ndarray): AR coefficient
            - lambduh (N ndarray): latent factor loadings
            - sigma2_x (double): white noise variance

    """
    # Inherit docstrings
    __doc__ += Likelihood.__doc__

    prior = {
            "mu_A0": 0.0,
            "sigma2_A0": 100.0,
            "mu_lambduh0": 0.0,
            "sigma2_lambduh0": 100.0,
            "alpha_x0": 3.0,
            "beta_x0": 2.0,
            }
    name = "AR"

    def __init__(self, data, sample_prior = False, **kwargs):
        raise NotImplementedError("This no longer works properly due to refactorization")
        self.y = data.get_matrix()[0]

        self._set_prior()
        self._default_parameter = {
                "A": 0.9 * np.ones(np.shape(self.y)[0]),
                "lambduh": np.ones(np.shape(self.y)[0]),
                "sigma2_x": 1.0,
                }
        self._set_default_parameters(**kwargs)
        self.init_parameters(sample_prior)
        return

    def _sample_from_prior(self):
        N = np.shape(self.y)[0]
        logger.warning("Sampling from prior is not recommended")
        parameter = {
                "A": np.random.normal(
                    loc=self.prior.mu_A0,
                    scale=np.sqrt(self.prior.sigma2_A0),
                    size=N),
                "lambduh": np.random.normal(
                    loc=self.prior.mu_lambduh0,
                    scale=np.sqrt(self.prior.sigma2_lambduh0),
                    size=N),
                "sigma2_x": 1.0/np.random.gamma(
                    shape=self.prior.alpha_x0,
                    scale=self.prior.beta_x0,
                    size=1)[0],
                }
        return parameter

    def loglikelihood(self, index, theta):
        u = self.y[index, 1:] - self.parameter.A[index] * self.y[index, 0:-1]
        mean = self.parameter.lambduh[index] * theta[1:]
        variance = self.parameter.sigma2_x
        loglikelihood = -0.5*((u - mean)/variance).dot(u - mean)
        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        posterior = theta_parameter
        for s_index in subset_indices:
            s_mean = np.append(
                    np.zeros(1),
                    (self.y[s_index, 1:] -
                        self.parameter.A[s_index] * self.y[s_index, 0:-1]) /
                    self.parameter.lambduh[s_index])
            s_variance = (self.parameter.sigma2_x /
                    (self.parameter.lambduh[s_index] ** 2))
            posterior = (posterior +
                    NaturalParameters(mean=s_mean, variance=s_variance))

        u_mean = self.parameter.lambduh[index] * posterior.get_mean()[1:]
        u_variance = (self.parameter.sigma2_x +
                self.parameter.lambduh[index] ** 2 *
                posterior.get_variance()[1:])
        u = self.y[index, 1:] - self.parameter.A[index] * self.y[index, 0:-1]
        loglikelihood = -0.5*((u - u_mean) / u_variance).dot(u - u_mean)
        return loglikelihood

    def moment(self, index, theta_parameter):
        mean = np.append(
                np.zeros(1),
                (self.y[index, 1:] -
                self.parameter.A[index] * self.y[index, 0:-1]) /
                self.parameter.lambduh[index])
        variance = (self.parameter.sigma2_x /
                (self.parameter.lambduh[index] ** 2))
        posterior = (theta_parameter +
                NaturalParameters(mean=mean,
                    variance=variance))
        return posterior

    def sample(self, indices, prior_parameter):
        posterior = prior_parameter
        for s_index in indices:
            s_mean = np.append(
                    np.zeros(1),
                    (self.y[s_index, 1:] -
                    self.parameter.A[s_index] * self.y[s_index, 0:-1]) /
                    self.parameter.lambduh[s_index])
            s_variance = (self.parameter.sigma2_x /
                    (self.parameter.lambduh[s_index] ** 2))
            posterior = (posterior +
                    NaturalParameters(mean=s_mean, variance=s_variance))
        theta_mean = posterior.get_mean()
        theta_variance = posterior.get_variance()
        return np.random.randn(posterior.T)*np.sqrt(theta_variance) + theta_mean

    def update_parameters(self, z, theta, parameter_name = None):
        if(parameter_name is None):
            self._update_A(z, theta)
            self._update_lambduh(z, theta)
            self._update_sigma2_x(z, theta)

        elif(parameter_name == "A"):
            self._update_A(z, theta)

        elif(parameter_name == "lambduh"):
            self._update_lambduh(z, theta)

        elif(parameter_name == "sigma2_x"):
            self._update_sigma2_x(z, theta)

        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)

        return

    def _update_A(self, z, theta):
        """ Update the AR coefficients A """
        N, T = np.shape(self.y)

        for index in range(0, N):
            x = self.y[index]
            lambduh = self.parameter.lambduh[index]
            eta = theta[z[index]]
            sigma2_x = self.parameter.sigma2_x

            # Calculate posterior natural parameter (from sufficient statistics)
            precision = self.prior.sigma2_A0 ** -1
            mean_precision = self.prior.mu_A0 * precision

            ## TODO: Handle t = 0
            ## (for now assume sigma2_0 is very large, so precision is zero)

            precision_t = (x[0:(T-1)] ** 2)/sigma2_x
            mean_precision_t = (x[1:T] - lambduh * eta[1:T])*x[0:(T-1)]/sigma2_x

            precision += np.sum(precision_t)
            mean_precision += np.sum(mean_precision_t)

            # Update A[index] by sampling from posterior
            sampled_A = np.random.normal(loc = mean_precision / precision,
                                         scale = precision ** -0.5)
            # Restrict A <= 0.999
            if sampled_A > 0.999:
                logger.warning(
                        "Sampled AR coefficient A is %f, which is > 0.999",
                        sampled_A)
                sampled_A = 0.999
            self.parameter.A[index] = sampled_A
            # Restrict A >= -0.999
            if sampled_A < -0.999:
                logger.warning(
                        "Sampled AR coefficient A is %f, which is < -0.999",
                        sampled_A)
                sampled_A = -0.999
            self.parameter.A[index] = sampled_A

        return

    def _update_lambduh(self, z, theta):
        """ Update the factor loading lambduh """
        N, T = np.shape(self.y)

        for index in range(0, N):
            x = self.y[index]
            a = self.parameter.A[index]
            eta = theta[z[index]]
            sigma2_x = self.parameter.sigma2_x

            # Calculate posterior natural parameter (from sufficient statistics)
            precision = self.prior.sigma2_lambduh0 ** -1
            mean_precision = self.prior.mu_lambduh0 * precision

            ## TODO: Handle t = 0
            ## (for now assume sigma2_0 is very large, so precision is zero)

            precision_t = (eta[1:T] ** 2)/sigma2_x
            mean_precision_t = (x[1:T] - a * x[0:(T-1)])*eta[1:T]/sigma2_x

            precision += np.sum(precision_t)
            mean_precision += np.sum(mean_precision_t)

            # Update lambduh[index] by sampling from posterior
            self.parameter.lambduh[index] = np.random.normal(
                    loc = mean_precision / precision,
                    scale = precision ** -0.5)

        return

    def _update_sigma2_x(self, z, theta):
        N, T = np.shape(self.y)

        alpha_x = self.prior.alpha_x0
        beta_x = self.prior.beta_x0

        for index in range(0, N):
            x = self.y[index]
            a = self.parameter.A[index]
            lambduh = self.parameter.lambduh[index]
            eta = theta[z[index]]

            ## TODO: Handle t = 0
            ## (for now assume sigma2_0 is very large, so precision is zero)

            # Calculate posterior natural parameter (from sufficient statistics)
            alpha_x += (T-1)/2.0
            beta_x += np.sum((x[1:T] - a*x[0:(T-1)] - lambduh*eta[1:T])**2)/2


        # Update sigma2_x
        self.parameter.sigma2_x = 1.0/np.random.gamma(shape = alpha_x,
                                                      scale = 1.0/beta_x,
                                                      size = 1)[0]
        return


