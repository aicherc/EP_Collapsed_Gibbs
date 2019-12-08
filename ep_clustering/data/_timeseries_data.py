#!/usr/bin/env python
"""

Create TimeSeries Model Data

"""

import numpy as np
import pandas as pd
import logging
from ep_clustering._utils import (
        Map, fix_docs, convert_matrix_to_df, convert_df_to_matrix
        )
from ep_clustering.data._gibbs_data import (
        GibbsData, _categorical_sample
        )

# Author Information
__author__ = "Christopher Aicher"

# Modify the root logger
logger = logging.getLogger(name=__name__)

# TimeSeries Model Data
@fix_docs
class TimeSeriesData(GibbsData):
    """ Data for TimeSeries GibbsSampler

    Additional Attributes:
        df (pd.DataFrame): data frame with data with columns
            (observation, dimension, ...)
        observation_name (string): name of observation column in df

    Additional Methods:
        get_matrix(column_name)
        subset(indices)
    """
    def __init__(self, df, *args, **kwargs):
        df = df.sort_index()
        super(TimeSeriesData, self).__init__(df=df, *args, **kwargs)
        return

    def _validate_data(self):
        super(TimeSeriesData, self)._validate_data()
        if "df" not in self:
            raise ValueError("`df` must be defined for TimeSeriesData")
        if "observation_name" not in self:
            raise ValueError(
                    "`observation_name` must be defined for TimeSeriesData")

        if "observation" not in self.df.index.names:
            raise ValueError("row_index 'observation' not in df index")
        if "dimension" not in self.df.index.names:
                raise ValueError("col_index 'dimension' not in df index")
        if self.observation_name not in self.df.columns:
            raise ValueError("observation_name {0} not in df".format(
                observation_name))

    def get_matrix(self, column_name=None):
        """ Return mean and count matrix (observation x dim) of column_name"""
        if column_name is None:
            column_name = self.observation_name
        if column_name not in self.df.columns:
            raise ValueError("column_name {0} not in df".format(column_name))

        return convert_df_to_matrix(self.df, value_name = column_name,
                row_index="observation", col_index="dimension")

    def subset(self, indices):
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if len(indices) == 1:
            # Bug with Pandas when indices is length 1 w/ 1 observation
            subset_df = self.df.loc[
                    self.df.index.get_level_values('observation').isin(indices)
                ]
        else:
            subset_df = self.df.loc[
                    self.df.index.get_level_values('observation').isin(indices)
            ]

        subset_data = type(self)(**self.copy()) # Copy Self
        subset_data.df = subset_df
        subset_data.num_obs = \
                subset_df.index.get_level_values('observation').max() + 1
        subset_data.num_dims = \
                subset_df.index.get_level_values('dimension').max() + 1
        subset_data._validate_data()
        return subset_data


# TimeSeries Model Data Generation
class TimeSeriesDataGenerator(object):
    """ TimeSeries Model Data Generator

    Args:
        num_obs (int): number of observations
        num_dim (int): number of dimensions
        K (int): number clusters
        **kwargs (dict):
            `Cluster Proportion Probabilities`
                cluster_proportions (ndarray): cluster proportion probabilities
                    or
                proportion_prior (ndarray): parameter for Dirichlet prior
            `Cluster Parameter`
                sigma2_x (double): latent process noise variance (default 1.0)
            `Series-Specific Parameters`
                A (ndarray): AR coefficients (default 0.99 * np.ones(N))
                sigma2_y (ndarray): obs noise variance (default np.ones(N))
                lambduh (ndarray): latent factor loadings (default np.ones(N))
                x0 (ndarray): latent process initialization
            `Options`
                missing_obs (double or ndarray): probability of missing obs
                regression (boolean): whether to include dummy covariates
                covariate_coeff (ndarray, optional): regression covariates
                    must by num_dim by num_coeff

    Methods:
        generate_cluster_proportions(proportion_prior): cluster_proportions
        generate_data(): returns data
    """
    def __init__(self, num_obs, num_dim, K, **kwargs):
        self.num_obs = num_obs
        self.num_dim = num_dim
        self.K = K
        self._parse_param(**kwargs)
        if kwargs.get('regression', False):
            self.param.covariate_coeff = kwargs.get('covariate_coeff',
                    np.zeros((self.num_obs, 2)))
        return

    def _parse_param(self, **kwargs):
        # Defines self.param
        default = {
            'sigma2_x': 1.0,
            'A': None,
            'sigma2_y': None,
            'sigma2_theta': 1.0,
            'lambduh': None,
            'missing_obs': 0.0,
            'x_0': None,
            }

        for key, value in kwargs.items():
            if key in default.keys():
                default[key] = value

        param = Map(default)

        # Handle variable arg defaults
        if param.A is None:
            param.A = 0.99 * np.ones(self.num_obs)
        if param.lambduh is None:
            param.lambduh = np.ones(self.num_obs)
        if param.sigma2_y is None:
            param.sigma2_y = np.ones(self.num_obs)
        if param.x_0 is None:
            var_0 = param.sigma2_x * (1.0/(1.0 - param.A**2))
            param.x_0 = np.random.normal(0,1,self.num_obs)*np.sqrt(var_0)

        self.param = param
        return

    def generate_cluster_proportions(self, proportion_prior=None):
        if proportion_prior is not None:
            self.param.proportion_prior = proportion_prior
        if 'proportion_prior' not in self.param:
            self.param.proportion_prior = 100 * np.ones(self.K)

        cluster_proportions = np.random.dirichlet(
                alpha = self.param.proportion_prior, size=1)

        return cluster_proportions

    def generate_data(self):
        # Get Proportions
        if 'cluster_proportions' not in self.param:
            self.param.cluster_proportions = self.generate_cluster_proportions()

        # Generate Data
        z = np.array(
            [ _categorical_sample(probs=self.param.cluster_proportions)
            for i in range(0,self.num_obs)],
            dtype=int)

        x = np.zeros((self.num_dim, self.num_obs))
        y = np.zeros((self.num_dim, self.num_obs))
        theta = np.zeros((self.num_dim, self.K))

        x_t = self.param.x_0
        for t in range(0,self.num_dim):
            theta_t = np.random.normal(0,1,self.K)
            theta[t,:] = theta_t

            x_t = self.param.A * x_t
            x_t += (np.random.normal(0,1,self.num_obs) *
                    np.sqrt(self.param.sigma2_x))
            x_t += (self.param.lambduh *
                    _one_hot(z, self.K).dot(theta_t))
            x[t,:] = x_t
            y[t,:] = x_t + (np.random.normal(0,1,self.num_obs) *
                    np.sqrt(self.param.sigma2_y))
            if self.param.missing_obs > 0.0:
                missing = np.random.rand(self.num_obs) < self.param.missing_obs
                y[t,missing] = np.nan
        df = convert_matrix_to_df(y.T, observation_name = "y")

        # Add Regression + Covariates
        if 'covariate_coeff' in self.param:
            # TODO: REFACTOR THIS
            covariate_coeff = self.param.covariate_coeff
            num_coeff = covariate_coeff.shape[1]
            for ii in range(num_coeff):
                df['cov_{0}'.format(ii)] = np.random.normal(size=df.shape[0])
            df['y_resid'] = df['y'] + 0.0
            y_new = df.reset_index().apply(lambda row: row['y_resid'] +
                np.sum([
                    row['cov_{0}'.format(ii)] *
                    covariate_coeff[int(row['observation']), ii]
                    for ii in range(num_coeff)
                    ]), axis=1)
            df['y'] = y_new.values

        # Format Output
        self.param['x'] = x.T
        data = TimeSeriesData(
                df = df,
                observation_name = "y",
                theta = theta.T,
                z = z,
                num_obs = self.num_obs,
                num_dim = self.num_dim,
                K = self.K,
                parameters = self.param,
                )
        return data

def _one_hot(z, K):
    """ Convert z into a one-hot bit vector representation """
    z_one_hot = np.zeros((z.size, K))
    z_one_hot[np.arange(z.size), z] = 1
    return z_one_hot



# Example Script
if __name__ == "__main__":
    print("Example Create TimeSeries Model Data")
    data_generator = TimeSeriesDataGenerator(
            num_obs = 50,
            num_dim = 100,
            K = 3,
            sigma2_x = 0.01)
    my_data = data_generator.generate_data()

#EOF
