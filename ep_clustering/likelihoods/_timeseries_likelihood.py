#!/usr/bin/env python
"""
Timeseries Likelihood
"""

# Import Modules
import numpy as np
import pandas as pd
import logging
from ep_clustering._utils import fix_docs, convert_matrix_to_df
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.kalman_filter import KalmanFilter, _cpp_available
from ep_clustering.exp_family import DiagNormalFamily

## Try Importing C++ Implementation
if _cpp_available:
    from ep_clustering.kalman_filter.c_kalman_filter import CKalmanFilter

logger = logging.getLogger(name=__name__)

@fix_docs
class TimeSeriesLikelihood(Likelihood):
    """ Correlated Time Series Object

    Args:
        use_cpp (boolean): whether to use the C++ implementation of the
            Kalman Filter (default = False)
        use_old_update (boolean): whether to use the old (incorrect) EP update
            For comparison purposes only (default = False)
        **kwargs:
            - x (N by T ndarray) - latent time series
            - A (N ndarray) - AR Coefficients
            - lambduh (N ndarray) - Factor Loadings
            - sigma2_x (double) - Covariance of latent errors
            - sigma2_y (N ndarray) - Variance of observation errors

    Note that `x` is updated when sampling `theta` using `sample()`.
    """
    __doc__ += Likelihood.__doc__
    name = "TimeSeries"

    def __init__(self, data, use_cpp = False,
            use_old_update = False, **kwargs):
        self.y, self.y_count = data.get_matrix()
        self.num_dim = data.num_dim
        super(TimeSeriesLikelihood, self).__init__(data, **kwargs)

        if use_cpp:
            if _cpp_available:
                self.Filter = CKalmanFilter
                self.name = "CppTimeSeries"
            else:
                logger.warning("C++ Implementation not found.\n " +
                "Have you called 'python setup.py build_ext -i' from src/ ?")
                logger.info("Using Python Kalman Filter Implementation")
                self.Filter = KalmanFilter
        else:
            self.Filter = KalmanFilter

        self._use_cpp = use_cpp
        self._use_old_update = use_old_update
        return

    def _get_default_prior(self):
        theta_prior = DiagNormalFamily(
                num_dim=self.num_dim,
                precision=np.ones(self.num_dim)/100,
        )
        return theta_prior

    def _get_default_parameters(self):
        default_parameter = {
                    "x": np.zeros(np.shape(self.y)) * np.nan,
                    "A": 0.9 * np.ones(np.shape(self.y)[0]),
                    "lambduh": np.ones(np.shape(self.y)[0]),
                    "sigma2_x": 1.0,
                    "sigma2_y": np.ones(np.shape(self.y)[0]),
                    }
        return default_parameter

    def _get_default_parameters_prior(self):
        prior = {
                "mu_A0": 0.0,
                "sigma2_A0": 100.0,
                "mu_lambduh0": 0.0,
                "sigma2_lambduh0": 100.0,
                "alpha_x0": 3.0,
                "beta_x0": 2.0,
                "alpha_y0": 3.0,
                "beta_y0": 2.0,
                }
        return prior

    def _sample_from_prior(self):
        N = np.shape(self.y)[0]
        logger.warning("Sampling from prior is not recommended")
        parameter = {
                "x": np.zeros(np.shape(self.y)) * np.nan,
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
                "sigma2_y": 1.0/np.random.gamma(
                    shape=self.prior.alpha_y0,
                    scale=self.prior.beta_y0,
                    size=N),
                }
        return parameter

    def loglikelihood(self, index, theta):
        kalman = self.Filter(y=np.array([self.y[index]]),
                A=self.parameter.A[index],
                lambduh=self.parameter.lambduh[index],
                sigma2_x=self.parameter.sigma2_x,
                sigma2_y=self.parameter.sigma2_y[index],
                eta_mean=theta,
                eta_var=np.zeros(np.size(theta)),
                y_count=np.array([self.y_count[index]]),
                )

        loglikelihood = kalman.calculate_log_likelihood()
        return loglikelihood

    def cluster_loglikelihood(self, indices, theta_parameter):
        if theta_parameter is None:
            theta_parameter = self._get_default_prior()

        if len(indices) == 0:
            return 0.0

        kalman = self.Filter(
            y = np.vstack([self.y[indices]]),
            A = np.hstack([self.parameter.A[indices]]),
            lambduh = np.hstack([self.parameter.lambduh[indices]]),
            sigma2_x=self.parameter.sigma2_x,
            sigma2_y=np.hstack([self.parameter.sigma2_y[indices]]),
            eta_mean=theta_parameter.get_mean(),
            eta_var=theta_parameter.get_variance(),
            y_count=np.vstack(
                [self.y_count[indices]]),
            )

        loglikelihood = kalman.calculate_log_likelihood()
        return loglikelihood

    def collapsed(self, index, subset_indices, theta_parameter):
        if theta_parameter is None:
            theta_parameter = self._get_default_prior()
        kalman = self.Filter(
            y=np.vstack([self.y[index], self.y[subset_indices]]),
            A=np.hstack([self.parameter.A[index],
                self.parameter.A[subset_indices]]),
            lambduh=np.hstack([self.parameter.lambduh[index],
                self.parameter.lambduh[subset_indices]]),
            sigma2_x=self.parameter.sigma2_x,
            sigma2_y=np.hstack([self.parameter.sigma2_y[index],
                self.parameter.sigma2_y[subset_indices]]),
            eta_mean=theta_parameter.get_mean(),
            eta_var=theta_parameter.get_variance(),
            y_count=np.vstack(
                [self.y_count[index], self.y_count[subset_indices]]),
            )

        loglikelihood = kalman.calculate_cond_log_likelihood(i=0)
        return loglikelihood

    def ep_loglikelihood(self, index, theta_parameter):
        approx_loglikelihood = self.collapsed(index=index, subset_indices=[],
                theta_parameter=theta_parameter)
        return approx_loglikelihood

    def moment(self, index, theta_parameter):
        #kalman = self.Filter( # TO FIX need to implement Cython Version
        kalman = KalmanFilter(
                y=np.array([self.y[index]]),
                A=self.parameter.A[index],
                lambduh=self.parameter.lambduh[index],
                sigma2_x=self.parameter.sigma2_x,
                sigma2_y=self.parameter.sigma2_y[index],
                eta_mean=theta_parameter.get_mean(),
                eta_var=theta_parameter.get_variance(),
                y_count=np.array([self.y_count[index]]),
                )

        log_scaling_coef = self.ep_loglikelihood(index, theta_parameter)
        if self._use_old_update and not self._use_cpp:
            likelihood_mean, likelihood_variance = kalman._old_moment_eta()
        else:
            likelihood_mean, likelihood_variance = kalman.moment_eta()

        site = DiagNormalFamily.from_mean_variance(
                    mean=likelihood_mean,
                    variance=likelihood_variance,
                    )
        post_approx = theta_parameter + site
        post_approx.log_scaling_coef = log_scaling_coef
        return post_approx

    def sample(self, indices, prior_parameter):
        if len(indices) == 0:
            return np.random.randn(prior_parameter.num_dim)*np.sqrt(
                    prior_parameter.get_variance()) + prior_parameter.get_mean()
        kalman = self.Filter(
            y=self.y[indices],
            A=self.parameter.A[indices],
            lambduh=self.parameter.lambduh[indices],
            sigma2_x=self.parameter.sigma2_x,
            sigma2_y=self.parameter.sigma2_y[indices],
            eta_mean=prior_parameter.get_mean(),
            eta_var=prior_parameter.get_variance(),
            y_count=self.y_count[indices],
            )
        x = kalman.sample_x() # Updates x through sampling theta
        self.parameter.x[indices] = x.T
        sampled_theta = kalman.sample_eta(x = np.copy(x))
        return sampled_theta

    def predict(self, new_data, prior_parameter, num_samples):
        sample_ys = []
        indices = np.unique(new_data.df.index.get_level_values('observation'))

        y_reg = pd.Series(np.zeros(new_data.df.shape[0]),
                    index = _fix_pandas_index(new_data.df.index)
                )

        kalman = self.Filter(
            y=self.y[indices],
            A=self.parameter.A[indices],
            lambduh=self.parameter.lambduh[indices],
            sigma2_x=self.parameter.sigma2_x,
            sigma2_y=self.parameter.sigma2_y[indices],
            eta_mean=prior_parameter.get_mean(),
            eta_var=prior_parameter.get_variance(),
            y_count=self.y_count[indices],
            )

        for s in range(0, num_samples):
            y = kalman.sample_y(x = self.parameter.x[indices].T)
            y_df = convert_matrix_to_df(y.T, '_prediction')['_prediction']
            y_df.index = y_df.index.set_levels(indices, level='observation')
            y_filtered = (y_reg + y_df).dropna()

            sample_ys.append(y_filtered.values)

        sampled_y = np.array(sample_ys)

        return sampled_y

    def update_parameters(self, z, theta, parameter_name = None):
        if(parameter_name is None):
            self._update_A(z, theta)
            self._update_lambduh(z, theta)
            self._update_sigma2_y()
            self._update_sigma2_x(z, theta)

        elif(parameter_name == "A"):
            self._update_A(z, theta)

        elif(parameter_name == "lambduh"):
            self._update_lambduh(z, theta)

        elif(parameter_name == "sigma2_y"):
            self._update_sigma2_y()

        elif(parameter_name == "sigma2_x"):
            self._update_sigma2_x(z, theta)

        elif(parameter_name == "x"):
            raise ValueError("x is updated when sampling theta using `sample`")

        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)

        return

    def _update_A(self, z, theta):
        """ Update the AR coefficients A """
        N, T = np.shape(self.parameter.x)

        for index in range(0, N):
            x = self.parameter.x[index]
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

        N, T = np.shape(self.parameter.x)

        for index in range(0, N):
            x = self.parameter.x[index]
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
        N, T = np.shape(self.parameter.x)

        alpha_x = self.prior.alpha_x0
        beta_x = self.prior.beta_x0

        for index in range(0, N):
            x = self.parameter.x[index]
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

    def _update_sigma2_y(self):
        N, T = np.shape(self.parameter.x)

        for index in range(0, N):
            x = self.parameter.x[index]
            y = self.y[index]
            y_count = self.y_count[index]
            is_obs = y_count > 0

            # Calculate posterior natural parameter (from sufficient statistics)
            alpha_y = self.prior.alpha_y0
            beta_y = self.prior.beta_y0

            alpha_y += np.sum(y_count[is_obs]) / 2.0
            beta_y += np.sum((y[is_obs] - x[is_obs])**2 * y_count[is_obs]) / 2.0

            # Sample sigma2_y
            self.parameter.sigma2_y[index] = \
                1.0 / np.random.gamma(shape = alpha_y,
                                      scale = 1.0 / beta_y,
                                      size = 1)
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        raise RuntimeError("Do not use 'separate_likeparams' argument with timeseries_likelihood")




def _fix_pandas_index(index):
    # Only keep the 'observation' and 'dimension' index
    index = index.droplevel(
                    [name for name in index.names
                        if name not in ['observation', 'dimension'] ]
                    )
    return index
