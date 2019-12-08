#!/usr/bin/env python
"""
Timeseries Likelihood with Regression
"""

# Import Modules
import numpy as np
import pandas as pd
import logging
from ep_clustering._utils import (
        fix_docs, convert_matrix_to_df, convert_df_to_matrix)
from ep_clustering.likelihoods._likelihoods import Likelihood
from ep_clustering.likelihoods._timeseries_likelihood import (
        TimeSeriesLikelihood, _fix_pandas_index)
from ep_clustering.kalman_filter import KalmanFilter, _cpp_available

## Try Importing C++ Implementation
if _cpp_available:
    from ep_clustering.kalman_filter.c_kalman_filter import CKalmanFilter

logger = logging.getLogger(name=__name__)

@fix_docs
class TimeSeriesRegressionLikelihood(TimeSeriesLikelihood):
    """ Correlated Time Series with regression Object

    Args:
        covariate_names (list of string): name of regression covariates
            There are `H` covariates
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
            - covariate_coeff (N by H ndarray) - regression coefficients

    Attributes:
        residuals (pd.DataFrame) - columns are
            original - observed data
            _ts_resid - observed data - estimated time series
            _reg - estimated regression
            _reg_resid -

    Note that `x` is updated when sampling `theta` using `sample()`.
    """
    __doc__ += Likelihood.__doc__
    name = "TimeSeriesRegression"

    def __init__(self, data, covariate_names, **kwargs):
        if len(covariate_names) == 0:
            raise ValueError("No covariates listed in covariate_name")
        for name in covariate_names:
            if name not in data.df.columns:
                raise ValueError(
                        "covariate_name {0} not found in data.df".format(name)
                        )
        self.covariate_names = covariate_names

        self.covariates = data.df[covariate_names]
        self.residuals = pd.DataFrame(
                data = np.zeros((len(self.covariates.index), 4)),
                index = data.df.index.copy(),
                columns = ['original', '_ts_resid', '_reg', '_reg_resid'],
             )
        self.residuals['original'] = data.df[data.observation_name]
        self.residuals['_reg_resid'] = data.df[data.observation_name]
        super(TimeSeriesRegressionLikelihood, self).__init__(data, **kwargs)
        self._update_covariate_coeff()
        return

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
                "mu_coeff_mean_0": 0.0,
                "sigma2_coeff_mean_0": 1.0,
                "alpha_coeff_var_0": 3.0,
                "beta_coeff_var_0": 2.0,
                }
        return prior

    def _get_default_parameters(self):
        N = np.shape(self.y)[0]
        H = len(self.covariate_names)
        default_parameter = {
                "x": np.zeros(np.shape(self.y)),
                "A": 0.9 * np.ones(N),
                "lambduh": np.ones(N),
                "sigma2_x": 1.0,
                "sigma2_y": np.ones(N),
                "covariate_coeff": np.zeros((N, H)),
                "mu_coeff": np.zeros(H),
                "sigma2_coeff": np.ones(H),
                }
        return default_parameter

    def _sample_from_prior(self):
        N = np.shape(self.y)[0]
        H = length(self.covariate_names)
        logger.warning("Sampling from prior is not recommended")
        parameter = {
                "x": np.zeros(np.shape(self.y)),
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
                "mu_coeff": np.random.normal(
                    loc=self.prior.mu_coeff_mean_0,
                    scale=np.sqrt(self.prior.sigma2_coeff_mean_0),
                    size=H),
                "sigma2_coeff": 1.0/np.random.gamma(
                    shape=self.prior.alpha_coeff_var_0,
                    scale=self.prior.beta_coeff_var_0,
                    size=H),
                }

        parameter['covariate_coeff'] = np.random.normal(
                loc=parameter['mu_coeff'],
                scale=np.sqrt(parameter['sigma2_coeff']),
                size=(N, H))
        return parameter

    def predict(self, new_data, prior_parameter, num_samples):
        sample_ys = []
        indices = np.array(new_data.df.index.levels[0])

        # Regression Component
        covariates_df = new_data.df[self.covariate_names]
        def to_apply(row):
            coeff = self.parameter.covariate_coeff[row.index.to_numpy()[0][0]]
            row['_reg'] = np.dot(row[self.covariate_names], coeff)
            return(row)
        y_reg = covariates_df.groupby(level="observation").apply(to_apply)['_reg']
        y_reg.index = _fix_pandas_index(y_reg.index)

        # Time Series Component
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
            self._update_covariate_coeff()

        elif(parameter_name == "A"):
            self._update_A(z, theta)

        elif(parameter_name == "lambduh"):
            self._update_lambduh(z, theta)

        elif(parameter_name == "sigma2_y"):
            self._update_sigma2_y(z, theta)

        elif(parameter_name == "sigma2_x"):
            self._update_sigma2_x(z, theta)

        elif(parameter_name == "covariate_coeff"):
            self._update_covariate_coeff()

        elif(parameter_name == "mu_coeff"):
            raise ValueError("mu_coeff is updated sampling covariate_coeff")

        elif(parameter_name == "sigma2_coeff"):
            raise ValueError("sigma2_coeff is updated sampling covariate_coeff")

        elif(parameter_name == "x"):
            raise ValueError("x is updated when sampling theta using `sample`")

        else:
            raise ValueError("Unrecognized parameter_name: " + parameter_name)

        return

    def _update_covariate_coeff(self):
        # Update ts_resid
        self._update_ts_resid()

        # Update covariate coeff
        N, H = np.shape(self.parameter.covariate_coeff)
        for index in range(0, N):
            # X is times by covariates matrix
            X = self.covariates.loc[index].values
            # target is times by 1 vector
            target = (self.residuals['_ts_resid'].loc[index]).values
            sigma2_y = self.parameter.sigma2_y[index]
            coeff = self.parameter.covariate_coeff[index]

            # Prior
            precision_coeff = np.diag(self.parameter.sigma2_coeff ** -1.0)
            mean_precision_coeff = \
                    self.parameter.mu_coeff / self.parameter.sigma2_coeff

            # Add Likelihood
            XtX = X.T.dot(X)
            precision_coeff += XtX / sigma2_y
            mean_precision_coeff += np.dot(X.T, target) / sigma2_y

            # Sample Posterior
            L = np.linalg.cholesky(precision_coeff)
            self.parameter.covariate_coeff[index,:] = \
                    np.linalg.solve(L.T, np.random.normal(size = H) +
                            np.linalg.solve(L, mean_precision_coeff))

        # Update Hyperpriors
        self._update_mu_coeff()
        self._update_sigma2_coeff()

        # Update reg_resid
        self._update_reg_resid()
        return

    def _update_mu_coeff(self):
        N, H = np.shape(self.parameter.covariate_coeff)

        for index in range(0, H):
            coeff = self.parameter.covariate_coeff[:, index]

            precision = self.prior.sigma2_coeff_mean_0
            mean_precision = self.prior.mu_coeff_mean_0 * precision

            precision += N / self.parameter.sigma2_coeff[index]
            mean_precision += np.sum(coeff / self.parameter.sigma2_coeff[index])

            self.parameter.mu_coeff[index] = np.random.normal(
                    loc = mean_precision/precision,
                    scale = precision ** -0.5)
        return

    def _update_sigma2_coeff(self):
        N, H = np.shape(self.parameter.covariate_coeff)

        for index in range(0, H):
            # TODO: This can be vectorized
            coeff = self.parameter.covariate_coeff[:, index]
            coeff_mean = self.parameter.mu_coeff[index]

            alpha_sigma2_coeff = self.prior.alpha_coeff_var_0
            beta_sigma2_coeff = self.prior.beta_coeff_var_0

            alpha_sigma2_coeff += N/2.0
            beta_sigma2_coeff += np.sum((coeff - coeff_mean)**2) / 2.0

            self.parameter.sigma2_coeff[index] = \
                    1.0 / np.random.gamma(shape = alpha_sigma2_coeff,
                                          scale = 1.0/ beta_sigma2_coeff,
                                          size = 1)

        return

    def _update_ts_resid(self):
        # Update data.df['_ts_resid']
        x_df = convert_matrix_to_df(self.parameter.x, observation_name='_x')
        observed_data = self.residuals['original']
        observed_data.index = _fix_pandas_index(observed_data.index)
        _ts_resid = (observed_data - x_df['_x']).dropna()
        self.residuals['_ts_resid'] = _ts_resid.tolist()
        return

    def _update_reg_resid(self):
        # Update data.df['_reg_resid'] and y
        def to_apply(row):
            coeff = self.parameter.covariate_coeff[row.index.to_numpy()[0][0]]
            row['_reg'] = np.dot(row[self.covariate_names], coeff)
            return(row)

        self.residuals['_reg'] = \
            self.covariates.groupby(level="observation").apply(to_apply)['_reg']
        self.residuals['_reg_resid'] = \
            self.residuals['original'] - self.residuals['_reg']
        self.y = \
            convert_df_to_matrix(self.residuals, value_name = "_reg_resid")[0]
        return


