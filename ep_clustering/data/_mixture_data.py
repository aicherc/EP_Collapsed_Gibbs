#!/usr/bin/env python
"""

Create Mixture Model Data

"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import invwishart, gamma
from ep_clustering._utils import Map, fix_docs
from ep_clustering.data._gibbs_data import (
        GibbsData, _categorical_sample
        )

# Author Information
__author__ = "Christopher Aicher"

# Modify the root logger
logger = logging.getLogger(name=__name__)

# Mixture Model Data
@fix_docs
class MixtureData(GibbsData):
    """ Data for Mixture Model Sampler

    Must contain `num_obs` and `num_dim` attributes

    Additional Attributes:
        matrix (ndarray): num_obs by num_dim data matrix
    """
    def _validate_data(self):
        super(MixtureData, self)._validate_data()
        if "matrix" not in self:
            raise ValueError("`matrix` must be defined for MixtureData")
        if not isinstance(self.matrix, np.ndarray):
            raise TypeError("`matrix` must be an ndarray")
        if self.matrix.shape != (self.num_obs, self.num_dim):
            raise ValueError("matrix.shape must match num_obs num_dim")
        return

# Mixture Model Data Generation
class MixtureDataGenerator(object):
    """ Mixture Model Data Generator

    Args:
        num_obs (int): number of observations
        num_dim (int): number of dimensions
        K (int): number clusters
        component_type (string): name (see create_mixture_component)
        component_options (dict): optional kwargs args for
            create_mixture_component
        **kwargs (dict):
            `Cluster Proportion Probabilities
                cluster_proportions (ndarray): cluster proportion probabilities
                    or
                proportion_prior (ndarray): parameter for Dirichlet prior
            `Cluster Component Parameters`
                cluster_parameters (list of dict): parameters for component
                    or
                component_prior (dict): args for `generate_component_parameters`

    Examples:
        my_data = MixtureDataGenerator(num_obs=100, num_dim=2, K=3)
        my_data_2 = MixtureDataGenerator(num_obs=100, num_dim=2, K=3,
            component_type = "gaussian")

        my_data_3 = MixtureDataGenerator(num_obs=100, num_dim=2, K=3,
            component_prior = {'mean_sd': 10})

        my_data_4 = MixtureDataGenerator(num_obs=100, num_dim=1, K=10,
            component_parameters = [
                {'mean': np.array([10]), 'variance': np.array([1])},
                {'mean': np.array([-10]), 'variance': np.array([1])},
            ])

    Methods:
        generate_cluster_proportions(proportion_prior): cluster_proportions
        generate_cluster_parameters(component_prior): component_parameters
        generate_data(): returns data
    """
    def __init__(self, num_obs, num_dim, K, component_type = 'diag_gaussian',
            component_options = {}, **kwargs):
        self.num_obs = num_obs
        self.num_dim = num_dim
        self.K = K
        self.param = Map(
                component_type = component_type,
                component_options = component_options,
                **kwargs)
        self.component = create_mixture_component(component_type, num_dim,
                **component_options)
        return

    def generate_cluster_proportions(self, proportion_prior=None):
        if proportion_prior is not None:
            self.param.proportion_prior = proportion_prior
        if 'proportion_prior' not in self.param:
            self.param.proportion_prior = 100 * np.ones(self.K)

        cluster_proportions = np.random.dirichlet(
                alpha = self.param.proportion_prior, size=1)

        return cluster_proportions

    def generate_cluster_parameters(self, component_prior=None):
        if component_prior is not None:
            self.param.component_prior = component_prior

        cluster_parameters = \
                [self.component.sample_parameters(
                    self.param.get('component_prior', {})
                    ) for k in range(self.K)]

        return cluster_parameters

    def generate_data(self):
        """ Generate Data

        Returns:
            data (MixtureData)
        """
        # Get Proportions
        if 'cluster_proportions' not in self.param:
            self.param.cluster_proportions = self.generate_cluster_proportions()

        # Get Component Parameters
        if 'cluster_parameters' not in self.param:
            self.param.cluster_parameters = self.generate_cluster_parameters()
        else:
            self.param.cluster_parameters = [
                    Map(cluster_parameter)
                    for cluster_parameter in self.param.cluster_parameters
                    ]

        # Generate Data
        z = np.array(
            [ _categorical_sample(probs=self.param.cluster_proportions)
            for i in range(0,self.num_obs)],
            dtype=int)

        matrix = np.zeros((self.num_obs, self.num_dim))
        for ii, z_ii in enumerate(z):
            matrix[ii,:] = self.component.sample_observation(
                    self.param.cluster_parameters[z_ii])

        # Format Output
        data = MixtureData(
                matrix = matrix,
                z = z,
                num_obs = self.num_obs,
                num_dim = self.num_dim,
                K = self.K,
                parameters = self.param,
                )
        return data

def create_mixture_component(name, num_dim, **kwargs):
    """ Return Component Object
    Args:
        name (string): type of component
            'diag_gaussian' - Diagonal Gaussian
            'gaussian' - Gaussian
            'mix_diag_gaussian' - Mixture of Diagonal Gaussian
            'mix_gaussian' - Mixture of Gaussians
            'student_t' - Student-T
            'von_mises_fisher' - Von Mises Fisher Distribution
        num_dim (int): dimension of component
        **kwargs: additional args for Component constructor
    Returns:
        component (Component): mixture component object
    """
    if name.lower() == "diag_gaussian":
        return DiagGaussianComponent(num_dim)
    elif name.lower() == "gaussian":
        return GaussianComponent(num_dim)
    elif name.lower() == "mix_diag_gaussian":
        return MixDiagGaussianComponent(num_dim, **kwargs)
    elif name.lower() == "mix_gaussian":
        return MixGaussianComponent(num_dim, **kwargs)
    elif name.lower() == "student_t":
        return StudentTComponent(num_dim, **kwargs)
    elif name.lower() == "von_mises_fisher":
        return VonMisesFisherComponent(num_dim, **kwargs)
    else:
        raise ValueError("Unrecognized component name '{0}'".format(name))

class Component(object):
    """ Mixture Component Distribution Object
    Args:
        num_dim (int)
        **kwargs

    Methods:
        sample_parameters(prior): return Map of parameters
        sample_observation(parameters): returns ndarray sample
    """
    def __init__(self, num_dim, **kwargs):
        self.num_dim = num_dim
        self._process_kwargs(**kwargs)

    def _process_kwargs(self, **kwargs):
        return

    def sample_parameters(self, prior):
        """ Sample parameters
        Args:
            prior (dict): (optional)
        """
        raise NotImplementedError()

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
        Returns:
            obs (ndarray)
        """
        raise NotImplementedError()

@fix_docs
class DiagGaussianComponent(Component):
    def sample_parameters(self, prior={}):
        """ Sample parameters
        Args:
            prior (dict): (optional)
                mean_mean (double or ndarray): mean for mean
                mean_sd (double or ndarray): standard deviation for mean
                variance_alpha (double or ndarray):
                    shape parameter for inverse Gamma
                variance_beta (double or ndarray):
                    rate parameter for inverse Gamma
        """
        if not isinstance(prior, dict):
            raise TypeError("Prior must be dict not '{0}'".format(type(prior)))
        mean_mean = prior.get("mean_mean", 0.0)
        mean_sd = prior.get("mean_sd", 2.0)
        variance_alpha = prior.get("sd_alpha", 5.0)
        variance_beta = prior.get("sd_beta", 5.0)

        mean = np.random.normal(size = self.num_dim) * mean_sd + mean_mean
        variance = 1.0/np.random.gamma(
            shape = variance_alpha,
            scale = 1.0/variance_beta,
            size = self.num_dim,
            )
        parameters = Map(mean = mean, variance = variance)
        return parameters

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                mean (double or ndarray)
                variance (double or ndarray)
        Returns:
            obs (ndarray)
        """
        obs = parameters.mean + \
                np.random.normal(size = self.num_dim) * \
                np.sqrt(parameters.variance)
        return obs

@fix_docs
class GaussianComponent(Component):
    def sample_parameters(self, prior={}):
        """ Sample parameters
        Args:
            prior (dict): (optional)
                mean_mean (ndarray): mean for mean
                mean_sd (ndarray): standard deviation for mean
                cov_psi (ndarray): scale matrix parameter for inverse Wishart
                cov_nu (double): df parameter for inverse Wishart
        """
        if not isinstance(prior, dict):
            raise TypeError("Prior must be dict not '{0}'".format(type(prior)))
        mean_mean = prior.get("mean_mean", np.zeros(self.num_dim))
        mean_sd = prior.get("mean_sd", np.ones(self.num_dim))
        cov_psi = prior.get("cov_psi", np.eye(self.num_dim))
        cov_nu = prior.get("cov_nu", self.num_dim + 2)

        mean = np.random.normal(size = self.num_dim) * mean_sd + mean_mean
        cov = invwishart.rvs(df = cov_nu, scale = cov_psi)
        parameters = Map(mean = mean, cov = cov)
        return parameters

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                mean (ndarray)
                cov (ndarray)
        Returns:
            obs (ndarray)
        """
        if self.num_dim > 1:
            obs = np.random.multivariate_normal(parameters.mean, parameters.cov)
            return obs
        else:
            obs = parameters.mean + \
                np.random.normal(size = self.num_dim) * np.sqrt(parameters.cov)
            return obs

@fix_docs
class MixDiagGaussianComponent(Component):
    """ Mixture of Scale Normals """
    __doc__ += Component.__doc__
    def sample_parameters(self, prior):
        raise NotImplementedError(
                "MixDiagGaussianComponent requires parameters to be specified")

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                mean (ndarray): length num_dim
                variance (double or ndarray): length num_dim
                proportions (ndarray): length num_scale_components
                scales (ndarray): length num_scale_components
                    (!!Note!! that this scales the variance)
        Returns:
            obs (ndarray)
        """
        if np.size(parameters.proportions) != np.size(parameters.scales):
            raise ValueError(
                    "scale mixture proportion and scale size must match")
        scale = np.random.choice(parameters.scales, size=1,
                p=parameters.proportions)[0]
        obs = parameters.mean + \
                np.random.normal(size = self.num_dim) * \
                np.sqrt(parameters.variance * scale)
        return obs

@fix_docs
class MixGaussianComponent(Component):
    """ Mixture of Scale Normals """
    __doc__ += Component.__doc__
    def sample_parameters(self, prior):
        raise NotImplementedError(
                "MixGaussianComponent requires parameters to be specified"
                )

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                mean (ndarray): length num_dim
                cov (ndarray): shape (num_dim, num_dim)
                proportions (ndarray): length num_scale_components
                scales (ndarray): length num_scale_components, nonnegative
                    (!!Note!! that this scales the variance)
        Returns:
            obs (ndarray)
        """
        if np.size(parameters.proportions) != np.size(parameters.scales):
            raise ValueError(
                    "scale mixture proportion and scale size must match")
        scale = np.random.choice(parameters.scales, size=1,
                p=parameters.proportions)[0]
        if self.num_dim > 1:
            obs = np.random.multivariate_normal(parameters.mean,
                    parameters.cov*np.sqrt(scale))
        else:
            obs = parameters.mean + \
                np.random.normal(size = self.num_dim) * \
                np.sqrt(parameters.cov * scale)
        return obs

@fix_docs
class VonMisesFisherComponent(Component):
    """ Von Mises Fisher Distribution """
    def sample_parameters(self, prior):
        raise NotImplementedError(
                "VonMiseFisherComponent requires parameters to be specified"
                )

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                normalized_mean (ndarray): lenth num_dim, l2 norm = 1
                concentration (double): positive
        Returns:
            obs (ndarray)
        """
        from spherecluster import sample_vMF
        if not np.isclose(np.linalg.norm(parameters.normalized_mean), 1.0):
            raise ValueError("normalized_mean must have l2 norm = 1.0")
        if parameters.concentration < 1e-16:
            raise ValueError("concentration must be positive")
        obs = sample_vMF(
                mu=parameters.normalized_mean,
                kappa=parameters.concentration,
                num_samples=1)[0]
        return obs


@fix_docs
class StudentTComponent(Component):
    """ Student T Distribution """
    def sample_parameters(self, prior={}):
        """ Sample parameters
        Args:
            prior (dict): (optional)
                mean_mean (ndarray): mean for mean
                mean_sd (ndarray): standard deviation for mean
                cov_psi (ndarray): scale matrix parameter for inverse Wishart
                cov_nu (double): df parameter for inverse Wishart
                df_alpha (double): shape for Gamma
                df_beta (double): rate for Gamma
        """
        if not isinstance(prior, dict):
            raise TypeError("Prior must be dict not '{0}'".format(type(prior)))
        mean_mean = prior.get("mean_mean", np.zeros(self.num_dim))
        mean_sd = prior.get("mean_sd", np.ones(self.num_dim))
        cov_psi = prior.get("cov_psi", np.eye(self.num_dim))
        cov_nu = prior.get("cov_nu", self.num_dim + 2)
        df_alpha = prior.get("df_alpha", 8.0)
        df_beta = prior.get("df_beta", 4.0)

        mean = np.random.normal(size = self.num_dim) * mean_sd + mean_mean
        cov = invwishart.rvs(df = cov_nu, scale = cov_psi)
        df = gamma.rvs(a=df_alpha, scale=1.0/df_beta)
        parameters = Map(mean=mean, cov=cov, df=df)
        return parameters

    def sample_observation(self, parameters):
        """ Sample observations
        Args:
            parameters (Map):
                mean (ndarray): vector of mean
                cov (ndarray): matrix of covariance
                df (double): positive degrees of freedom for t-distribution
        Returns:
            obs (ndarray)
        """
        u = gamma.rvs(a=parameters.df/2.0, scale=2.0/parameters.df)
        scaled_cov = parameters.cov / u

        if self.num_dim > 1:
            obs = np.random.multivariate_normal(parameters.mean, scaled_cov)
            return obs
        else:
            obs = parameters.mean + \
                np.random.normal(size = self.num_dim) * np.sqrt(scaled_cov)
            return obs
        return obs


# Example Script
if __name__ == "__main__":
    print("Example Create Mixture Model Data")
    data_generator = MixtureDataGenerator(
            num_obs = 30,
            num_dim = 2,
            K = 3)
    my_data = data_generator.generate_data()


#EOF
