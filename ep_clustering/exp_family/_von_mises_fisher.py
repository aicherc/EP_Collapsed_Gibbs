#!/usr/bin/env python
"""
Von Mises Fisher Exponential Family Helper Class
"""

# Import Modules
import numpy as np
import logging
import scipy.stats
from ep_clustering._utils import fix_docs, logsumexp
from ._exponential_family import ExponentialFamily
from spherecluster import sample_vMF
from scipy.special import iv

logger = logging.getLogger(name=__name__)
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )


# VonMisesFisher Exponential Family
@fix_docs
class VonMisesFisherFamily(ExponentialFamily):
    """ Von Mises Fisher Site Approximation Class

    This is the likelihood for the mean, treating the concentration as fixed.

    Natural Parameters:
        'mean': (num_dim ndarray)

    Helper Functions:
        get_normalized_mean(): return mean
        get_concentration(): return variance
    """
    def __init__(self, num_dim, log_scaling_coef=0.0, mean=None):
        # Set default mean
        if mean is None:
            #mean = np.ones(num_dim)/np.sqrt(num_dim) * count
            mean = np.zeros(num_dim)

        # Check Dimensions
        if np.size(mean) != num_dim:
            raise ValueError("mean must be size num_dim")

        super(VonMisesFisherFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                mean=mean,
                )
        return

    def is_valid_density(self):
        if np.linalg.norm(self.natural_parameters['mean']) < 1e-16:
            return False
        return True

    def sample(self):
        self._check_is_valid_density()
        sample = sample_vMF(
                mu=self.get_normalized_mean(),
                kappa=self.get_concentration(),
                num_samples=1)[0]
        return sample

    def logpartition(self):
        self._check_is_valid_density()
        concentration = self.get_concentration()
        order = (0.5 * self.num_dim - 1)
        log_partition = 0.5 * self.num_dim * np.log(2*np.pi)
        log_partition -= order * np.log(concentration)
        log_partition += amos_asymptotic_log_iv(order, concentration)
        return log_partition

    def get_normalized_mean(self):
        """ Return mean a.k.a. mu (on unit ball) """
        mean = self.natural_parameters['mean']
        norm = np.linalg.norm(mean)
        if norm > 0:
            normalized_mean =  mean / norm
        else:
            raise RuntimeError("mean is zero-vector")
        return normalized_mean

    def get_concentration(self):
        """ Return concentration a.k.a. kappa """
        concentration = np.linalg.norm(self.natural_parameters['mean'])
        return concentration


# VonMisesFisherProdGamma Exponential Family
@fix_docs
class VonMisesFisherProdGammaFamily(ExponentialFamily):
    """ Von Mises Fisher Product Gamma Site Approximation Class

    This is the likelihood for the mean and concentration parameter of the vMF.

    Natural Parameters:
        'mean': (num_dim ndarray) mean parameter for vMF mean
        'alpha_minus_one': (double) shape parameter for vMF concentration
        'beta': (double) rate parameter for vMF scale

    Helper Functions:
        get_normalized_mean(): return mean
    """
    def __init__(self, num_dim, log_scaling_coef=0.0, mean=None,
            alpha_minus_one=0.0, beta=0.0):
        # Set default mean
        if mean is None:
            #mean = np.ones(num_dim)/np.sqrt(num_dim) * count
            mean = np.zeros(num_dim)

        # Check Dimensions
        if np.size(mean) != num_dim:
            raise ValueError("mean must be size num_dim")

        super(VonMisesFisherProdGammaFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                mean=mean,
                alpha_minus_one=alpha_minus_one,
                beta=beta,
                )
        return

    def _get_concentration_quantiles(self, breaks=20):
        q = (np.arange(0, breaks) + 0.5)/(breaks*1.0)*0.98 + 0.01
        q = np.concatenate((
            np.logspace(-5, -2, 5),
            q,
            1.0-np.logspace(-5, -2, 5)[::-1],
            ))
        alpha = self.natural_parameters['alpha_minus_one'] + 1.0
        beta = self.natural_parameters['beta']
        kappas = scipy.stats.gamma.ppf(q=q, a=alpha, scale=1.0/beta)
        return kappas

    def _get_concentration_quantile_weights(self, kappas):
        alpha = self.natural_parameters['alpha_minus_one'] + 1.0
        beta = self.natural_parameters['beta']
        mid_points = (kappas[1:]+kappas[:-1])/2.0
        cdf = np.concatenate(
                (np.zeros(1),
                 scipy.stats.gamma.cdf(x=mid_points, a=alpha, scale=1.0/beta),
                 np.ones(1),
                 )
            )
        weights = cdf[1:] - cdf[:-1]
        return weights

    def _get_concentration_logpartitions(self, kappas):
        order = (0.5 * self.num_dim - 1)
        log_partitions = np.zeros_like(kappas)
        log_partitions += 0.5 * self.num_dim * np.log(2*np.pi)
        log_partitions -= order * np.log(kappas)
        log_partitions += amos_asymptotic_log_iv(order, kappas)
        return log_partitions

    def is_valid_density(self):
        if np.linalg.norm(self.natural_parameters['mean']) < 1e-16:
            return False
        if (self.natural_parameters['alpha_minus_one'] + 1.0) < 1e-16:
            return False
        if self.natural_parameters['beta'] < 1e-16:
            return False
        return True

    def sample(self):
        self._check_is_valid_density()

        alpha = self.natural_parameters['alpha_minus_one'] + 1.0
        beta = self.natural_parameters['beta']
        sample_concentration = scipy.stats.gamma.rvs(
                a=alpha, scale=1.0/beta, size=1)[0]

        sample_mean = sample_vMF(
                mu=self.get_normalized_mean(),
                kappa=sample_concentration,
                num_samples=1)[0]

        return dict(
                mean=sample_mean,
                concentration=sample_concentration,
                )

    def logpartition(self):
        self._check_is_valid_density()
        kappas = self._get_concentration_quantiles()
        weights = self._get_concentration_quantile_weights(kappas)
        log_partitions = self._get_concentration_logpartitions(
                kappas * np.linalg.norm(self.natural_parameters['mean'])
                )

        log_partition = logsumexp(log_partitions, weights)
        return log_partition

    def get_normalized_mean(self):
        """ Return mean a.k.a. mu (on unit ball) """
        mean = self.natural_parameters['mean']
        norm = np.linalg.norm(mean)
        if norm > 0:
            normalized_mean =  mean / norm
        else:
            raise RuntimeError("mean is zero-vector")
        return normalized_mean

    def get_mean_variance_concentration(self):
        """ Return mean and variance of concentration """
        alpha = self.natural_parameters['alpha_minus_one'] + 1.0
        beta = self.natural_parameters['beta']

        mean = alpha/beta
        variance = alpha/(beta**2)

        return dict(mean=mean, variance=variance)

# Helper Functions
def hankle_asymptotic_log_iv(order, z):
    """ Asymptotic Expansion of the modified Bessel function

    Asymptotic form by Hankle (see http://dlmf.nist.gov/10.40)
    Using the first three terms
    (see https://en.wikipedia.org/wiki/Bessel_function#Asymptotic_forms)

    Do not use! (Only good for small order + very large z, z >> order**2)

    """
    log_iv = z + 0.0
    log_iv -= 0.5 * np.log(2*np.pi*z)
    log_iv += np.log(1 +
            -(4*order**2 -1)/(8*z) +
            (4*order**2-1)*(4*order**2-9)/(2*(8*z)**2) +
            -(4*order**2-1)*(4*order**2-9)*(4*order**2-25)/(6*(8*z)**3))
    return log_iv

def amos_asymptotic_log_iv(order, z):
    """ Asymptotic Expansion of the modified Bessel function

    Asymptotic form by spherecluster using `_log_H_asymptotic`
    (See utility function implementation notes in movMF.R from
    https://cran.r-project.org/web/packages/movMF/index.html)

    """
    # The approximation from spherecluster is good up to a constant
    log_iv = np.log(iv(order, 100)) - (
        _log_H_asymptotic(order, 100) + order*np.log(100)
        )
    log_iv += _log_H_asymptotic(order, z) + order*np.log(z)
    return log_iv

def _log_H_asymptotic(nu, kappa):
    """Compute the Amos-type upper bound asymptotic approximation on H where
    log(H_\nu)(\kappa) = \int_0^\kappa R_\nu(t) dt.

    See "lH_asymptotic <-" in movMF.R and utility function implementation notes
    from https://cran.r-project.org/web/packages/movMF/index.html
    """
    beta = (nu + 0.5)
    kappa_l = np.min(np.array([kappa,
        np.sqrt((3. * nu + 11. / 2.) * (nu + 3. / 2.))*np.ones_like(kappa)]),
        axis=0)
    return (
        _S(kappa, nu + 0.5, beta) +
        (_S(kappa_l, nu, nu + 2.) - _S(kappa_l, nu + 0.5, beta))
    )


def _S(kappa, alpha, beta):
    """Compute the antiderivative of the Amos-type bound G on the modified
    Bessel function ratio.

    See "S <-" in movMF.R and utility function implementation notes from
    https://cran.r-project.org/web/packages/movMF/index.html
    """
    kappa = 1. * np.abs(kappa)
    alpha = 1. * alpha
    beta = 1. * np.abs(beta)
    a_plus_b = alpha + beta
    u = np.sqrt(kappa**2 + beta**2)
    if alpha == 0:
        alpha_scale = 0
    else:
        alpha_scale = alpha * np.log((alpha + u) / a_plus_b)

    return u - beta - alpha_scale


