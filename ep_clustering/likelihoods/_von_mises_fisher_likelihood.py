#!/usr/bin/env python
"""
Von Mises Fisher Likelihood
"""

# Import Modules
import numpy as np
import scipy.special
from scipy.optimize import root
import logging
from ep_clustering._utils import fix_docs, logsumexp
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.likelihoods._slice_sampler import SliceSampler
from ep_clustering.exp_family._von_mises_fisher import (
        VonMisesFisherFamily,
        VonMisesFisherProdGammaFamily,
        amos_asymptotic_log_iv,
    )
from spherecluster import sample_vMF
MAX_CONCENTRATION = 10.0**9
MIN_CONCENTRATION = 10**-3

logger = logging.getLogger(name=__name__)
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )

@fix_docs
class FixedVonMisesFisherLikelihood(Likelihood):
    """ Von Mises Fisher Likelihood with fixed concentration

    Args:
        concentration_update (string): method for updating concentration
            "map": (default) use the MAP estimator
            "slice_sampler": slow
        num_slice_steps (int): number of slice sampler steps
        **kwargs:
            concentration (double) - concentration (a.k.a. kappa)
    """
    # Inherit Docstrings
    __doc__ += Likelihood.__doc__

    # Class Variables
    name = "FixedVonMisesFisher"

    def __init__(self, data, concentration_update="map",
            num_slice_steps=5, **kwargs):
        self.y = data.matrix
        self.num_dim = data.num_dim
        super(FixedVonMisesFisherLikelihood, self).__init__(data, **kwargs)

        self.concentration_update = concentration_update
        self.num_slice_steps = num_slice_steps
        return

    def deepcopy(self):
        """ Return a copy """
        other = type(self)(data = self.data,
                concentration_update=self.concentration_update,
                num_slice_steps=self.num_slice_steps,
                theta_prior=self.theta_prior)
        other.parameter = self.parameter.deepcopy()
        other.prior = self.prior.deepcopy()
        return other

    def _get_default_prior(self):
        theta_prior = VonMisesFisherFamily(
                num_dim = self.num_dim,
                mean=np.ones(self.num_dim)/np.sqrt(self.num_dim) * 1e-9)
        return theta_prior

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        default_parameter = {
            "concentration": 1.0,
            }

        return default_parameter

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        prior = {
            "alpha_concentration0": 2.0,
            "beta_concentration0": 0.1,
            }
        return prior

    def _sample_from_prior(self):
        parameter = {
                    "concentration": 1.0/np.random.gamma(
                        shape=self.prior.alpha_concentration0,
                        scale=self.prior.beta_concentration0,
                        size=1)
                    }
        return parameter

    def loglikelihood(self, index, theta):
        y_index = self.y[index]
        order = (0.5 * self.num_dim - 1)
        loglikelihood = self.parameter.concentration * theta.dot(y_index) + \
                order * np.log(self.parameter.concentration) + \
                -0.5*self.num_dim*np.sqrt(2*np.pi) + \
                -amos_asymptotic_log_iv(order, self.parameter.concentration)
        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        loglikelihood = 0.0

        cavity_posterior = theta_parameter
        for s_index in subset_indices:
            s_y = self.y[s_index]
            cavity_posterior = (cavity_posterior + VonMisesFisherFamily(
                        num_dim=self.num_dim,
                        mean=s_y*self.parameter.concentration,
                    ))
        loglikelihood -= cavity_posterior.logpartition()


        y = self.y[index]
        likelihood = VonMisesFisherFamily(
                num_dim=self.num_dim,
                mean=y*self.parameter.concentration,
                )
        loglikelihood -= likelihood.logpartition()

        posterior = cavity_posterior + likelihood
        loglikelihood += posterior.logpartition()
        return loglikelihood

    def moment(self, index, theta_parameter):
        y_index = self.y[index]
        site = VonMisesFisherFamily(
                num_dim=self.num_dim,
                mean=y_index * self.parameter.concentration,
                )
        unnormalized_post_approx = (theta_parameter + site)
        unnormalized_post_approx.log_scaling_coef = \
                unnormalized_post_approx.logpartition() - \
                (theta_parameter.logpartition() + site.logpartition())
        return unnormalized_post_approx

    def sample(self, indices, prior_parameter):
        posterior = prior_parameter
        for index in indices:
            y_index = self.y[index]
            posterior = posterior + VonMisesFisherFamily(
                    num_dim=self.num_dim,
                    mean=y_index * self.parameter.concentration,
                    )
        return posterior.sample()

    def update_parameters(self, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_concentration(z, theta)
        elif parameter_name == "variance":
            self._update_concentration(z, theta)
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return

    def _update_concentration(self, z, theta, k_list=None):
        if k_list is None:
            k_list = range(np.shape(theta)[0])

        if self.concentration_update == "map":
            # MAP Estimator Update from
            # http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
            kappa, n = 0.0, 0.0
            for k in k_list:
                ind = (z == k)
                n_k = (np.sum(ind)*1.0)
                r_bar_k = np.linalg.norm(np.sum(self.y[ind,:], axis=0))/n_k
                r_bar_k *= (1-1e-6)
                kappa_k = (r_bar_k*self.num_dim - r_bar_k**3)/(1.0 - r_bar_k**2)
                if kappa_k > MAX_CONCENTRATION:
                    kappa_k = MAX_CONCENTRATION
                kappa += n_k * kappa_k
                n += n_k
            self.parameter.concentration = kappa/n
            if n == 0:
                self.parameter.concentration = MIN_CONCENTRATION
            if (np.isinf(self.parameter.concentration) or
                    np.isnan(self.parameter.concentration)):
                raise ValueError("concentration is invalid")

        elif self.concentration_update == "slice_sampler":
            # Slice Sampler Update
            logprior = lambda kappa: scipy.stats.gamma.logpdf(
                    kappa, a=self.prior.alpha_concentration0,
                    scale=1.0/self.prior.beta_concentration0,
                    )
            n = 0.0
            mu_T_x = 0.0
            for k in k_list:
                ind = (z == k)
                n += (np.sum(ind)*1.0)
                mu_T_x += np.dot(theta[k], np.sum(self.y[ind,:], axis=0))
            order = self.num_dim/2.0 - 1.0

            def logf(kappa):
                logf = logprior(kappa)
                logf += kappa * mu_T_x
                logf += n * order * np.log(kappa)
                logf -= n * amos_asymptotic_log_iv(order, kappa)
                return logf

            slice_sampler = SliceSampler(
                    logf=logf, lower_bound=0.0,
                    num_steps=self.num_slice_steps)
            self.parameter.concentration = slice_sampler.sample(
                    x_init = self.parameter.concentration,
                    )
            if (np.isinf(self.parameter.concentration) or
                    np.isnan(self.parameter.concentration)):
                raise ValueError("concentration is invalid")
        else:
            raise NotImplementedError(
                "Unrecognized `concentration_update`={0}".format(
                    self.concentration_update,
                    ))
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_concentration(z, theta, k_list=[k])
        elif parameter_name == "concentration":
            self._update_concentration(z, theta, k_list=[k])
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return


@fix_docs
class VonMisesFisherLikelihood(Likelihood):
    """ Von Mises Fisher Likelihood

    Args:
        moment_update (string):
            'exact' - use root finding to match sufficient statistics
            'variance' - use algebra to match first two moments (faster)
        decay_factor (double):
            decay factor for posterior moment natural parameters
        breaks (int): number of points used in numerical integration

        **kwargs:
    """
    # Inherit Docstrings
    __doc__ += Likelihood.__doc__

    # Class Variables
    name = "VonMisesFisher"

    def __init__(self, data, moment_update='exact', decay_factor=1.0, breaks=20,
            **kwargs):
        self.y = data.matrix
        self.num_dim = data.num_dim
        self.moment_update = moment_update
        self.decay_factor = decay_factor
        if not isinstance(breaks, int):
            raise TypeError("breaks must be an int")
        self.breaks = breaks
        super(VonMisesFisherLikelihood, self).__init__(data, **kwargs)
        return

    def deepcopy(self):
        """ Return a copy """
        other = type(self)(data = self.data,
                moment_update=self.moment_update,
                decay_factor=self.decay_factor,
                breaks=self.breaks,
                theta_prior=self.theta_prior)
        other.parameter = self.parameter.deepcopy()
        other.prior = self.prior.deepcopy()
        return other

    def _get_default_prior(self):
        theta_prior = VonMisesFisherProdGammaFamily(
                num_dim = self.num_dim,
                mean=np.ones(self.num_dim)/np.sqrt(self.num_dim) * 1e-9,
                alpha_minus_one=1.0,
                beta=0.1,
                )
        return theta_prior

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        default_parameter = {}
        return default_parameter

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        prior = {}
        return prior

    def _sample_from_prior(self):
        parameter = {}
        return parameter

    def loglikelihood(self, index, theta):
        y_index = self.y[index]
        order = (0.5 * self.num_dim - 1)
        loglikelihood = theta['concentration'] * theta['mean'].dot(y_index) + \
                order * np.log(theta['concentration']) + \
                -0.5*self.num_dim*np.sqrt(2*np.pi) + \
                -amos_asymptotic_log_iv(order, theta['concentration'])
        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        raise NotImplementedError("collapsed likelihood not implemented")

    def ep_loglikelihood(self, index, theta_parameter):
        approx_loglikelihood = 0.0
        y_index = self.y[index]

        cavity_posterior = theta_parameter
        kappas = cavity_posterior._get_concentration_quantiles(
                breaks=self.breaks)
        weights = cavity_posterior._get_concentration_quantile_weights(kappas)

        site_logpart = cavity_posterior._get_concentration_logpartitions(kappas)
        cavity_logpart = cavity_posterior._get_concentration_logpartitions(
                kappas * np.linalg.norm(
                    cavity_posterior.natural_parameters['mean']
                    )
                )
        post_approx_logpart = cavity_posterior._get_concentration_logpartitions(
                kappas * np.linalg.norm(
                    y_index + cavity_posterior.natural_parameters['mean']
                    )
                )

        approx_loglikelihood = logsumexp(
                post_approx_logpart - site_logpart - cavity_logpart,
                weights)
        return approx_loglikelihood

    def moment(self, index, theta_parameter):
        y_index = self.y[index]

        kappas = theta_parameter._get_concentration_quantiles(
                breaks=self.breaks)
        weights = theta_parameter._get_concentration_quantile_weights(kappas)

        site_logpart = theta_parameter._get_concentration_logpartitions(kappas)
        cavity_logpart = theta_parameter._get_concentration_logpartitions(
                kappas * np.linalg.norm(
                    theta_parameter.natural_parameters['mean']
                    )
            )
        post_approx_logpart = theta_parameter._get_concentration_logpartitions(
                kappas * np.linalg.norm(
                    y_index + theta_parameter.natural_parameters['mean']
                    )
                )
        logparts = post_approx_logpart - site_logpart - cavity_logpart

        # Calculate Sufficient Statistic Moments
        logpartition = logsumexp(logparts, weights)
        mean_kappa = np.exp(
                logsumexp(logparts, weights * kappas) -
                logpartition
                )
        mean_kappa_2 = np.exp(
                logsumexp(logparts, weights * kappas**2) -
                logpartition
                )
        var_kappa = mean_kappa_2 - mean_kappa**2
        if np.isnan(mean_kappa) or mean_kappa < 0:
            raise ValueError("Invalid Mean_Kappa")

        # Convert Moments to Alpha + Beta
        if self.moment_update == 'exact':
            mean_log_kappa = np.exp(
                    logsumexp(logparts, weights * np.log(kappas)) -
                    logpartition
                    )

            beta0 = mean_kappa / var_kappa
            alpha0 = mean_kappa * beta0
            def fun(x):
                return (scipy.special.digamma(x) - np.log(x) +
                        np.log(mean_kappa) - mean_log_kappa)
            alpha = root(fun, alpha0).x[0]
            beta = alpha/mean_kappa
        elif self.moment_update == 'variance':
            beta = mean_kappa / var_kappa
            alpha = mean_kappa * beta
        else:
            raise ValueError("Unrecognized moment_update `{0}`".format(
                self.moment_update))

        # Apply Decay Factor
        if self.decay_factor < 1.0:
            alpha_minus_one_diff = (alpha - 1) - \
                    theta_parameter.natural_parameters['alpha_minus_one']
            beta_diff = beta - \
                    theta_parameter.natural_parameters['beta']
            alpha = (self.decay_factor * alpha_minus_one_diff) + 1 + \
                    theta_parameter.natural_parameters['alpha_minus_one']
            beta = (self.decay_factor * beta_diff) + \
                    theta_parameter.natural_parameters['beta']

        # Return post approx
        unnormalized_post_approx = theta_parameter.copy()
        unnormalized_post_approx.natural_parameters['mean'] += y_index
        unnormalized_post_approx.natural_parameters['alpha_minus_one'] = \
                (alpha - 1.0) * self.decay_factor
        unnormalized_post_approx.natural_parameters['beta'] = \
                beta * self.decay_factor
        unnormalized_post_approx.log_scaling_coef = logpartition
        return unnormalized_post_approx

    def sample(self, indices, prior_parameter):
        raise NotImplementedError("sample theta not implemented")

    def update_parameters(self, z, theta, parameter_name = None):
        if parameter_name is not None:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        if parameter_name is not None:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return



