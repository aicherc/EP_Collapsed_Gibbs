#!/usr/bin/env python
"""
Student T Likelihood
"""

# Import Modules
import numpy as np
import scipy
import logging
from scipy.stats import wishart, gamma
from scipy.special import gammaln
from scipy.optimize import brentq
from ep_clustering._utils import fix_docs, logsumexp
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.exp_family._mean_variance import (
        NormalWishartFamily, multidigamma
    )

logger = logging.getLogger(name=__name__)
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )


@fix_docs
class StudentTLikelihood(Likelihood):
    """ Student T Likelihood Object

    Args:
        moment_update (string):
            'exact' - use root finding to match sufficient statistics
        breaks (int): number of points used in numerical integration

        **kwargs :
            df (double) - degrees of freedom (default = 3.0)
            u (N ndarray) - auxiliary scale parameter for t-distribution
    """
    # Inherit Docstrings
    __doc__ += Likelihood.__doc__

    # Class Variables
    name = "StudentT"

    def __init__(self, data, moment_update='exact', breaks=20, **kwargs):
        self.y = data.matrix
        self.num_dim = data.num_dim
        self.moment_update = moment_update
        if not isinstance(breaks, int):
            raise TypeError("breaks must be an int")
        self.breaks = breaks
        super(StudentTLikelihood, self).__init__(data, **kwargs)
        return

    def deepcopy(self):
        """ Return a copy """
        other = type(self)(data = self.data,
                moment_update=self.moment_update,
                breaks=self.breaks,
                theta_prior=self.theta_prior)
        other.parameter = self.parameter.deepcopy()
        other.prior = self.prior.deepcopy()
        return other

    def _get_default_prior(self):
        theta_prior = NormalWishartFamily(num_dim = self.num_dim,
                kappa = 0.1,
                nu_minus = 1.0,
                zeta=np.eye(self.num_dim)*(self.num_dim),
                )
        return theta_prior

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        default_parameter = {
            "df": 3.0,
            "u": np.ones(np.shape(self.y)[0]),
            }
        return default_parameter

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        prior = {"df_prior": 3.0}
        return prior

    def _sample_from_prior(self):
        parameter = {
                "df": self.prior.df_prior,
                "u": np.random.gamma(
                        shape=self.prior.df_prior/2.0,
                        scale=2.0/self.prior.df_prior,
                        size=self.num_dim)
                    }
        return parameter

    def loglikelihood(self, index, theta):
        y_index = self.y[index]

        # Conditioned on mean, precision, and u_index -> likelihood is Gaussian
        mean = theta['mean']
        precision = theta['precision']
        u_index = self.parameter.u[index]

        scaled_precision = u_index * precision

        loglikelihood = -0.5*self.num_dim*np.log(2*np.pi) + \
            -0.5*(y_index-mean).dot(scaled_precision.dot(y_index-mean)) + \
            0.5*self.num_dim*np.log(u_index) + \
            0.5*np.linalg.slogdet(precision)[1]

        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        kappa, nu, mu, psi = theta_parameter.get_kappa_nu_mu_psi()

        y_indices = self.y[subset_indices]
        u_indices = self.parameter.u[subset_indices]
        uy_indices = np.outer(u_indices, np.ones(self.num_dim)) * y_indices
        u_sum = np.sum(u_indices)
        uy_sum = np.sum(uy_indices, axis=0)
        uyy_sum = y_indices.T.dot(uy_indices)

        kappa_post = kappa + u_sum
        nu_post = nu + len(subset_indices)
        mu_post = (kappa * mu + uy_sum)/(kappa + u_sum)
        psi_post = psi + uyy_sum + (
            kappa*np.outer(mu, mu) - kappa_post*np.outer(mu_post, mu_post)
            )

        y_index = self.y[index]
        u_index = self.parameter.u[index]

        # Generalization of t-Distribution predictive posterior
        loglikelihood = -0.5*self.num_dim*np.log(np.pi) + \
            0.5*self.num_dim*np.log(u_index) + \
            0.5*self.num_dim*(np.log(kappa_post)-np.log(kappa_post+u_index)) + \
            scipy.special.gammaln((nu_post + 1.0)/2.0) + \
            -scipy.special.gammaln((nu_post - self.num_dim + 1.0)/2.0) + \
            -0.5*np.linalg.slogdet(psi_post)[1] + \
            -0.5*(nu_post + 1.0)*np.log(1 +
                (kappa_post*u_index)/(kappa_post + u_index) *
                (y_index-mu_post).dot(
                    np.linalg.solve(psi_post, y_index-mu_post))
                )

        return loglikelihood

    def _get_u_grid(self, recalculate=False):
        # Create Grid of numerical values for moment_update
        # Grid is based on Gamma(alpha=df/2, beta=df/2)
        if not hasattr(self, "_u_grid") or recalculate:
            # Define _u_grid
            q = (np.arange(0, self.breaks) + 0.5)/(self.breaks*1.0)*0.98 + 0.01
            q = np.concatenate((
                np.logspace(-5, -2, 5),
                q,
                1.0-np.logspace(-5, -2, 5)[::-1],
                ))
            alpha = self.parameter.df/2.0
            beta = self.parameter.df/2.0
            self._u_grid = gamma.ppf(q=q, a=alpha, scale=1.0/beta)

            # Define _u_weights
            mid_points = (self._u_grid[1:]+self._u_grid[:-1])/2.0
            cdf = np.concatenate((
                np.zeros(1),
                gamma.cdf(x=mid_points, a=alpha, scale=1.0/beta),
                np.ones(1),
                ))
            self._u_weights = cdf[1:] - cdf[:-1]

        return self._u_grid, self._u_weights

    def moment(self, index, theta_parameter):
        # Need to numpy vectorize this
        y_index = self.y[index]
        kappa, nu, mu, psi = theta_parameter.get_kappa_nu_mu_psi()

        # Helper constants
        V = np.linalg.inv(psi)
        logdetpsi = np.linalg.slogdet(psi)[1]
        delta = y_index - mu
        V_delta = np.linalg.solve(psi, delta)
        VdeltadeltaTV = np.outer(V_delta, V_delta)
        deltaTVdelta = delta.dot(V_delta)

        u_grid, u_weights = self._get_u_grid()

        # Calculate loglikelihood
        logpartitions = 0.5*self.num_dim*np.log(u_grid/np.pi) + \
                0.5*self.num_dim*np.log(kappa) + \
                -0.5*self.num_dim*np.log(kappa+u_grid) + \
                gammaln((nu+1.0)/2.0) - gammaln((nu-self.num_dim+1.0)/2.0) + \
                -0.5*logdetpsi + \
                -0.5*(nu+1.0)*np.log(
                    1.0 + (kappa*u_grid)/(kappa+u_grid) * deltaTVdelta
                    )
        logpartition = logsumexp(logpartitions, u_weights)

        # Calculate Moments of Titled Distribution
        expect_mean_precision_mean = 0.0
        expect_mean_precision = np.zeros(self.num_dim)
        expect_precision = np.zeros((self.num_dim, self.num_dim))
        expect_log_precision = (
                multidigamma((nu+1.0)/2.0, self.num_dim) + \
                self.num_dim * np.log(2) - logdetpsi
                )

        moment_weights = u_weights * np.exp(logpartitions-logpartition)
        for u, weight in zip(u_grid, moment_weights):
            psi_post_inv = (
                V - (kappa * u)/(kappa + u) * (VdeltadeltaTV) / \
                (1 + (kappa * u)/(kappa + u) * deltaTVdelta)
                )
            mu_post = (kappa * mu + u * y_index)/(kappa + u)

            expect_mean_precision_mean += weight * (
                    (nu+1) * mu_post.dot(psi_post_inv.dot(mu_post)) +
                    self.num_dim/(kappa + u)
                    )
            expect_mean_precision += weight * (nu+1) * psi_post_inv.dot(mu_post)
            expect_precision += weight * (nu+1) * psi_post_inv

            expect_log_precision += weight * (
                    -1.0 * np.log(1 + (kappa * u)/(kappa+u) * deltaTVdelta)
                    )

        # Project into NormalWishartFamily
        if self.moment_update == 'exact':
            constant = (expect_log_precision - self.num_dim * np.log(2) - \
                    np.linalg.slogdet(expect_precision)[1])
            d = self.num_dim
            def fun(x):
                obj = multidigamma(x/2.0, d) - d*np.log(x) - constant
                return obj
            nu_proj = brentq(fun, d, (nu+1)*10.0)
            L_V_proj = np.linalg.cholesky(expect_precision / nu_proj)
            psi_proj = np.linalg.solve(L_V_proj.T,
                    np.linalg.solve(L_V_proj, np.eye(self.num_dim)))
            mu_proj = np.linalg.solve(expect_precision, expect_mean_precision)
            kappa_proj = self.num_dim * (expect_mean_precision_mean - \
                    mu_proj.dot(expect_mean_precision))**-1.0
        else:
            raise ValueError("Unrecognized moment_update `{0}`".format(
                self.moment_update))

        unnormalized_post_approx = NormalWishartFamily.from_mu_kappa_nu_psi(
                mu=mu_proj, kappa=kappa_proj,
                nu=nu_proj, psi=psi_proj,
                )
        unnormalized_post_approx.log_scaling_coef = logpartition
        return unnormalized_post_approx


    def ep_loglikelihood(self, index, theta_parameter):
        # Copied from self.moment()
        y_index = self.y[index]
        kappa, nu, mu, psi = theta_parameter.get_kappa_nu_mu_psi()

        # Helper constants
        logdetpsi = np.linalg.slogdet(psi)[1]
        delta = y_index - mu
        deltaTVdelta = delta.dot(np.linalg.solve(psi, delta))

        u_grid, u_weights = self._get_u_grid()

        # Calculate loglikelihood
        logpartitions = 0.5*self.num_dim*np.log(u_grid/np.pi) + \
                0.5*self.num_dim*np.log(kappa) + \
                -0.5*self.num_dim*np.log(kappa+u_grid) + \
                gammaln((nu+1.0)/2.0) - gammaln((nu-self.num_dim+1.0)/2.0) + \
                -0.5*logdetpsi + \
                -0.5*(nu+1.0)*np.log(
                    1.0 + (kappa*u_grid)/(kappa+u_grid) * deltaTVdelta
                    )

        logpartition = logsumexp(logpartitions, u_weights)
        return logpartition

    def sample(self, indices, prior_parameter):
        kappa, nu, mu, psi = prior_parameter.get_kappa_nu_mu_psi()

        y_indices = self.y[indices]
        u_indices = self.parameter.u[indices]
        uy_indices = np.outer(u_indices, np.ones(self.num_dim)) * y_indices
        u_sum = np.sum(u_indices)
        uy_sum = np.sum(uy_indices, axis=0)
        uyy_sum = y_indices.T.dot(uy_indices)

        kappa_post = kappa + u_sum
        nu_post = nu + len(indices)
        mu_post = (kappa * mu + uy_sum)/(kappa + u_sum)
        psi_post = psi + uyy_sum + (
            kappa*np.outer(mu, mu) - kappa_post*np.outer(mu_post, mu_post)
            )

        precision = wishart(df=nu_post,
                            scale=np.linalg.inv(psi_post)).rvs()
        L = np.linalg.cholesky(precision) * \
                np.sqrt(kappa_post)
        mean = mu_post + np.linalg.solve(L.T,
                np.random.normal(size=self.num_dim))

        return dict(mean=mean, precision=precision)

    def update_parameters(self, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_u(z, theta)
        elif parameter_name == "u":
            self._update_u(z, theta)
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return

    def _update_u(self, z, theta, k=None):
        df = self.parameter.df

        for ii, z_i in enumerate(z):
            if (k is None) or (z_i == k):
                # Condition on Y_i
                delta = self.y[ii] - theta[z_i]['mean']
                alpha_u = df/2.0 + self.num_dim/2.0
                beta_u = df/2.0 + 0.5 * np.dot(
                        delta, np.dot(theta[z_i]['precision'], delta)
                        )
                self.parameter.u[ii] = gamma(a=alpha_u, scale=1.0/beta_u).rvs()
            else:
                # Sample from Prior
                self.parameter.u[ii] = gamma(a=df/2.0, scale=2.0/df).rvs()
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        if parameter_name is None:
            self._update_u(z, theta, k=k)
        elif parameter_name == "u":
            self._update_u(z, theta, k=k)
        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)
        return



# EOF
