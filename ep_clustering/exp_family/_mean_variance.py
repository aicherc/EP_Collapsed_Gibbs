#!/usr/bin/env python
"""
Natural Parameters Helper for Mean + Variance Exp Families
"""

# Import Modules
import numpy as np
from ep_clustering._utils import fix_docs
from ._exponential_family import ExponentialFamily
from scipy.special import digamma, multigammaln
from scipy.stats import invwishart, wishart

# Author Information
__author__ = "Christopher Aicher"

# Helper function for Inverse Wishart
def multidigamma(a, d):
    """ Returns psi_d(a), where psi_d is the multivariate digamma of d

    Calculation exploits the recursion:
        psi_d(a) = \sum_{i=1}^d psi_1(a + (1-i)/2)
    where psi_1 is the digamma function
    """
    if np.isscalar(a):
        return np.sum(digamma(a - 0.5*np.arange(d)),
            axis=-1)
    else:
        return np.sum(digamma(a[...,None] - 0.5*np.arange(d)),
            axis=-1)



@fix_docs
class NormalInverseWishartFamily(ExponentialFamily):
    """ Normal Inverse Wishart Family Site Approximation Class

    Natural Parameters:
        lambduh (double):
        mean_lambduh (num_dim ndarray):
        nu (double):
        psi_plus (num_dim by num_dim ndarray):

    Helper Functions:
        get_mean(): return expected mean
        get_mean_precision(): return expected mean-precision
        get_variance(): return expected variance
        get_precision(): return expected precision
        get_log_variance(): return expected log variance
    """
    def __init__(self, num_dim, log_scaling_coef=0.0,
            lambduh=0.0, mean_lambduh=None,
            nu=0.0, psi_plus=None):
        # Set Default mean_precision and precision
        if mean_lambduh is None:
            mean_lambduh = np.zeros(num_dim)
        if psi_plus is None:
            psi_plus = np.zeros((num_dim, num_dim))

        # Check Dimensions
        if np.size(mean_lambduh) != num_dim:
            raise ValueError("mean_lambduh must be size num_dim")
        if np.shape(psi_plus) != (num_dim, num_dim):
            if np.size(psi_plus) == num_dim:
                psi_plus = np.diag(psi_plus)
            else:
                raise ValueError("psi_plus must be size num_dim by num_dim")

        super(NormalInverseWishartFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                lambduh=lambduh,
                mean_lambduh=mean_lambduh,
                nu=nu,
                psi_plus=psi_plus,
                )
        return

    def get_mean(self):
        """ Return the expected mean """
        if not self.is_valid_density():
            return np.NaN
        mean = self.natural_parameters['mean_lambduh'] / \
                self.natural_parameters['lambduh']
        return mean

    def get_mean_precision(self):
        """ Return the expected mean precision """
        if not self.is_valid_density():
            return np.NaN
        mean_precision = self.natural_parameters['nu'] * np.linalg.solve(
                self._get_psi(), self.get_mean())
        return mean_precision

    def get_variance(self):
        """ Return the expected variance (returns NaN if variance DNE)"""
        if not self.is_valid_density():
            return np.NaN
        nu = self.natural_parameters['nu']
        if nu - self.num_dim - 1 <= 0:
            variance = np.NaN
        else:
            variance = self._get_psi() / (nu - self.num_dim - 1)
        return variance

    def get_precision(self):
        """ Return the expected precision """
        if not self.is_valid_density():
            return np.NaN
        precision = self.natural_parameters['nu'] * \
                np.linalg.inv(self._get_psi())
        return precision

    def get_log_variance(self):
        """ Return the expected log det of variance """
        if not self.is_valid_density():
            return np.NaN
        log_det_psi = np.linalg.slogdet(self._get_psi())[1]
        log_det_var = log_det_psi - self.num_dim * np.log(2) - \
            multidigamma(a=self.natural_parameters['nu']/2.0, p=self.num_dim)
        return log_det_var

    def _get_psi(self):
        psi = self.natural_parameters['psi_plus'] - \
                np.outer(self.natural_parameters['mean_lambduh'],
                self.natural_parameters['mean_lambduh']) / \
                self.natural_parameters['lambduh']
        return psi

    def is_valid_density(self):
        if self.natural_parameters['lambduh'] < 1e-12:
            return False
        if self.natural_parameters['nu'] < self.num_dim - 1 + 1e-12:
            return False
        eigvals = np.linalg.eigvals(self._get_psi())
        if np.any(eigvals < 1e-12):
            return False
        return True

    def sample(self):
        self._check_is_valid_density()
        variance = invwishart(df=self.natural_parameters['nu'],
                scale=self._get_psi()).rvs()
        mean = np.random.multivariate_normal(mean=self.get_mean(),
                cov=variance/self.natural_parameters['lambduh'])
        return dict(mean=mean, variance=variance)


@fix_docs
class NormalWishartFamily(ExponentialFamily):
    """ Normal Wishart Family Site Approximation Class

    Canonical Parameters:
        mu (num_dim ndarray):
        psi (num_dim by num_dim ndarray):
        kappa (double): pseudo count on mu observations
        nu (double): pseudo count on psi observations

    Natural Parameters:
        kappa (double): kappa
        nu_minus (double): nu - num_dim
        eta (num_dim ndarray): kappa * mu
        zeta (num_dim by num_dim ndarray): psi + kappa * mu * mu.T

    Helper Functions:
        get_mean(): return expected mean
        get_mean_precision(): return expected mean-precision
        get_variance(): return expected variance
        get_precision(): return expected precision
        get_log_variance(): return expected log variance
    """
    def __init__(self, num_dim, log_scaling_coef=0.0,
            kappa=0.0, nu_minus=0.0, eta=None, zeta=None):
        # Set Default mean_precision and precision
        if eta is None:
            eta = np.zeros(num_dim)
        if zeta is None:
            zeta = np.zeros((num_dim, num_dim))

        # Check Dimensions
        if np.size(eta) != num_dim:
            raise ValueError("mean_lambduh must be size num_dim")
        if np.shape(zeta) != (num_dim, num_dim):
            if np.size(zeta) == num_dim:
                zeta = np.diag(zeta)
            else:
                raise ValueError("zeta must be size num_dim by num_dim")

        super(NormalWishartFamily, self).__init__(
                num_dim=num_dim,
                log_scaling_coef=log_scaling_coef,
                kappa=kappa,
                nu_minus=nu_minus,
                eta=eta,
                zeta=zeta,
                )
        return

    @staticmethod
    def from_mu_kappa_nu_psi(mu, kappa, nu, psi):
        """ Helper function for constructing from canonical parameters """
        num_dim = np.size(mu)
        if np.shape(psi) != (num_dim, num_dim):
            raise ValueError("psi must be size num_dim by num_dim")
        eta = kappa * mu
        zeta = psi + kappa * np.outer(mu, mu)
        return NormalWishartFamily(
                num_dim=num_dim,
                kappa=kappa,
                nu_minus=nu - num_dim,
                eta=eta,
                zeta=zeta,
                )

    def get_mean(self):
        """ Return the expected mean """
        if not self.is_valid_density():
            return np.NaN
        mean = self.natural_parameters['eta'] / \
                self.natural_parameters['kappa']
        return mean

    def get_mean_precision(self):
        """ Return the expected mean precision """
        if not self.is_valid_density():
            return np.NaN
        mean_precision = self._get_nu() * np.linalg.solve(
                self._get_psi(), self.get_mean())
        return mean_precision

    def get_variance(self):
        """ Return the expected variance (returns NaN if variance DNE)"""
        if not self.is_valid_density():
            return np.NaN
        nu = self._get_nu()
        if nu - self.num_dim - 1 <= 0:
            variance = np.NaN
        else:
            variance = self._get_psi() / (nu - self.num_dim - 1)
        return variance

    def get_precision(self):
        """ Return the expected precision """
        if not self.is_valid_density():
            return np.NaN
        precision = self._get_nu() * \
                np.linalg.inv(self._get_psi())
        return precision

    def get_log_precision(self):
        """ Return the expected log det of precision """
        if not self.is_valid_density():
            return np.NaN
        log_det_psi = np.linalg.slogdet(self._get_psi())[1]
        log_det_precision = -log_det_psi + self.num_dim * np.log(2) + \
            multidigamma(a=self._get_nu()/2.0, p=self.num_dim)
        return log_det_precision

    def get_log_variance(self):
        """ Return the expected log det of variance """
        if not self.is_valid_density():
            return np.NaN
        log_det_var = -1.0 * self.get_log_variance()
        return log_det_var

    def get_kappa_nu_mu_psi(self):
        """ Return Canonical parametrization """
        kappa = self.natural_parameters['kappa']
        mu = self.natural_parameters['eta'] / \
                self.natural_parameters['kappa']
        nu = self._get_nu()
        psi = self._get_psi()
        return [kappa, nu, mu, psi]

    def _get_psi(self):
        psi = self.natural_parameters['zeta'] - \
                np.outer(self.natural_parameters['eta'],
                self.natural_parameters['eta']) / \
                self.natural_parameters['kappa']
        return psi

    def _get_nu(self):
        return self.natural_parameters['nu_minus'] + self.num_dim

    def is_valid_density(self):
        if self.natural_parameters['kappa'] < 1e-12:
            return False
        if self.natural_parameters['nu_minus'] < 1e-12:
            return False
        eigvals = np.linalg.eigvals(self._get_psi())
        if np.any(eigvals < 1e-12):
            return False
        return True

    def logpartition(self):
        self._check_is_valid_density()
        nu = self._get_nu()
        psi = self._get_psi()
        kappa = self.natural_parameters['kappa']
        log_partition = 0.5 * self.num_dim * np.log(2*np.pi)
        log_partition += 0.5 * self.num_dim * nu * np.log(2)
        log_partition -= 0.5 * nu * np.linalg.slogdet(psi)[1]
        log_partition -= 0.5 * self.num_dim * np.log(kappa)
        log_partition += multigammaln(nu/2.0, self.num_dim)
        return log_partition

    def sample(self):
        self._check_is_valid_density()
        precision = wishart(df=self._get_nu(),
                            scale=np.linalg.inv(self._get_psi())).rvs()
        L = np.linalg.cholesky(precision) * \
                np.sqrt(self.natural_parameters['kappa'])
        mean = self.get_mean() + \
                np.linalg.solve(L.T, np.random.normal(size=self.num_dim))
        return dict(mean=mean, precision=precision)




