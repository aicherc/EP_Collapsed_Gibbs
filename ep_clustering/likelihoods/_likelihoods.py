#!/usr/bin/env python
"""
Likelihood Objects for Gibbs Sampler
"""

# Import Modules
import logging
import ep_clustering._utils as _utils

logger = logging.getLogger(name=__name__)

# Author Information
__author__ = "Christopher Aicher"

# Abstract Class for Likelihood
class Likelihood(object):
    """ Likelihood function for GibbsSampler

    Likelihood function of theta

    Args:
        data (GibbsData): the data
        prior (ExponentialFamily): the prior for theta
        sample_prior (bool): whether to initalize parameters from prior
        **kwargs (kwargs): additional likelihood arguments
            likelihood_prior_kwargs,
            default_likelihood_parameters,

    Attributes:
        theta_prior (ExponentialFamily): prior for theta
        parameter (Map): likelihood parameters
        prior (Map): likelihood prior parameters
        name (string): name of likelihood function

    Methods:
        - loglikelihood(index, theta): return loglikelihood
        - init_parameters(sample_prior, **kwargs): initialize parameters
        - collapsed(index, subset_indices, theta_parameter):
            return conditional loglikelihood
        - ep_loglikelihood(index, theta): return ep approx to loglikelihood
        - moment(index, theta_parameter): return moment for EP
        - sample(indices, prior_parameter): return sample from posterior
        - predict(indices, prior_parameter, num_samples, **kwargs): returns
            posterior predictions for observations
        - update_parameters(z, theta): update parameters via Gibbs sampling
    """
    name = "Likelihood"
    def __init__(self, data, theta_prior=None, **kwargs):
        self.data = data
        if theta_prior is None:
            theta_prior = self._get_default_prior()
        self.theta_prior = theta_prior

        kwargs = self._set_parameters_prior(**kwargs)
        kwargs = self._set_default_parameters(**kwargs)
        self.init_parameters(sample_prior=kwargs.pop("sample_prior",False))
        self._process_kwargs(**kwargs)
        return

    def deepcopy(self):
        """ Return a copy """
        other = type(self)(data = self.data, theta_prior=self.theta_prior)
        other.parameter = self.parameter.deepcopy()
        other.prior = self.prior.deepcopy()
        return other

    def _get_default_prior(self):
        """ Returns default theta prior ExponentialFamily object """
        raise NotImplementedError()

    def _get_default_parameters(self):
        """Returns default parameters dict"""
        raise NotImplementedError()

    def _get_default_parameters_prior(self):
        """Returns default parameters prior dict"""
        raise NotImplementedError()

    def _set_parameters_prior(self, **kwargs):
        """ Set prior of likelihood parameters """
        prior = self._get_default_parameters_prior()
        for key in list(kwargs.keys()):
            if key in prior.keys():
                prior[key] = kwargs.pop(key) + 0.0
        self.prior = _utils.Map(prior)
        return kwargs

    def _set_default_parameters(self, **kwargs):
        """ Set default_parameter """
        default_parameter = self._get_default_parameters()
        for key in list(kwargs.keys()):
            if key in default_parameter.keys():
                default_parameter[key] = kwargs.pop(key) + 0.0
        self._default_parameter = _utils.Map(default_parameter)
        return kwargs

    def _process_kwargs(self, **kwargs):
        """ Process remaining kwargs """
        for key in kwargs.keys():
            logger.info("Ignoring unrecognized kwarg: {0}".format(key))
        return

    def _sample_from_prior(self):
        """ Sample initial likelihood parameter setting from prior """
        raise NotImplementedError()

    def init_parameters(self, sample_prior=False, **kwargs):
        """ Initialize the likelihood parameters

        Args:
            sample_prior (bool): whether to sample from prior or use defaults
            **kwargs:
                likelihood parameter initialization
        """
        if sample_prior:
            # Sample parameter from prior
            parameter = self._sample_from_prior()

        else:
            # Use default parameter
            parameter = self._default_parameter

        # Override any specified parameters
        for key, value in kwargs.items():
            if key in parameter.keys():
                parameter[key] = value + 0.0
            else:
                logging.info("Ignoring unrecognized kwarg: {0}".format(key))

        # Set parameter
        self.parameter = _utils.Map(parameter)
        return

    def loglikelihood(self, index, theta):
        """ Calculate the loglikelihood for a single series.

        Calculates the loglikelihood for y[index] for cluster parameter theta

        Args:
            index (int): index of series
            theta (ndarray): cluster parameter

        Returns:
            loglikelihood (double): loglikelihood of y[index] for theta
        """
        raise NotImplementedError()

    def cluster_loglikelihood(self, indices, theta_parameter):
        """ Calculate the collapsed loglikelihood for an entire cluster. """
        raise NotImplementedError()

    def collapsed(self, index, subset_indices, theta_parameter):
        """ Calculate the collapsed loglikelihood for a series

        Calculates the collapsed loglikelihood for y[index] for a cluster
        given the other series y[subset_indices] and prior theta_parameter

        Args:
            index (int): index of series
            subset_indices (list): indices of other series assigned to cluster
            theta_parameter (ExponentialFamily): cavity prior for cluster parameter

        Returns:
            loglikelihood (double): collapsed loglikelihood of y[index]
        """
        raise NotImplementedError()

    def ep_loglikelihood(self, index, theta_parameter):
        """ Calculate the EP approximation to the loglikelihood

        Calculate the log of the integral of the unnormalized
        tilted distribution (i.e. the log_scaling_coef)
        """
        approx_loglikelihood = self.moment(
                index=index,
                theta_parameter=theta_parameter,
                ).log_scaling_coef
        return approx_loglikelihood


    def moment(self, index, theta_parameter):
        """ Calculate the ExponentialFamily approx to the tilted distribution.

        Calculates the approximate moments and log_scaling_coef of
        the tilted distribution for theta_parameter's exponential family

        Args:
            index (int): index of series
            theta_parameter (ExponentialFamily):
                cavity distribution

        Returns:
            unnormalized_post_approx (ExponetialFamily):
        """
        raise NotImplementedError()

    def sample(self, indices, prior_parameter):
        """ Sample cluster parameter from posterior

        Samples cluster parameter theta conditioning on
        series in cluster y[indices] and prior

        Args:
            indices (list): indices of series in cluster
            prior_parameter (ExponentialFamily): prior for cluster parameter

        Returns:
            theta (T ndarray): random sample of cluster parameter from posterior
        """
        raise NotImplementedError()

    def predict(self, new_data, prior_parameter, num_samples, **kwargs):
        """ Sample observations from predictive posterior

        Args:
            new_data (GibbsData): new data to predict
            prior_parameter (ExponentialFamily): prior for cluster parameter
            num_samples (int): number of samples
            **kwargs: additional arguments

        Returns:
            sampled_y (np.ndarray): num_samples by num_rows(new_data)
        """
        raise NotImplementedError()
        return

    def update_parameters(self, z, theta, parameter_name = None):
        """ Update likelihood based on cluster assignments and parameters

        Args:
            z (num_obs ndarray):
                cluster assignments
            theta (K by num_dim ndarray):
                cluster parameters
            parameter_name(string, optional):
                parameter to update, default updates all parameters

        Returns:
            None: updates likelihood.parameter

        """
        raise NotImplementedError()
        return

    def update_local_parameters(self, k, z, theta, parameter_name = None):
        """ Update likelihood parameters based only on cluster k

        Args:
            k (int): cluster
            z (num_obs ndarray):
                cluster assignments
            theta (K by num_dim ndarray):
                cluster parameters
            parameter_name(string, optional):
                parameter to update, default updates all parameters

        Returns:
            None: updates likelihood.parameter

        """
        raise NotImplementedError()
        return



#EOF
