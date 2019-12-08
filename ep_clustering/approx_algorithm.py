#!/usr/bin/env python
"""

Approximate Algorithm Classes for Gibbs Sampler
"""

# Import Modules
import logging
import ep_clustering as ep_clustering
import numpy as np
from ep_clustering._utils import fix_docs, Map

logger = logging.getLogger(name=__name__)

# Author Information
__author__ = "Christopher Aicher"


# Construct Approx Algorithm
def construct_approx_algorithm(name, **kwargs):
    """ Construct ApproxAlgorithm by name

    Args:
        name (string): name of likelihood
            'naive': naive Gibbs
            'collapsed': collapsed Gibbs
            'EP': EP
        separate_likeparams (bool)
        **kwargs (kwargs): additional arguments (e.g. exp_family)

    Examples:
        ep_alg = construct_approx_algorithm('EP', exp_family=NormalFamily)

    Returns:
        approx_alg (ApproxAlgorithm): approximate algorithmm for sampling z
    """
    if(name == "naive"):
        return NaiveAlgorithm(**kwargs)
    elif(name == "collapsed"):
        return CollapsedAlgorithm(**kwargs)
    elif(name == "EP"):
        return EPAlgorithm(**kwargs)
    else:
        raise ValueError("Unrecognized name {0}".format(name))

# Abstract Class for Likelihood
class ApproxAlgorithm(object):
    """ Algorithm Class for GibbsSampler

    Args:
        **kwargs

    Attributes:
        name (string): name of algorithm
        parameters (Map): approximation parameters
        separate_likeparams (bool)

    Methods:
        init_approx(sampler): initializes the approx algorithm
        loglikelihood_z(index, state): return approx loglikelihoods of z[index]
        update_approx(index, old_z, new_z): update approximation for z[index]

    """
    name = "BaseClass"
    def __init__(self, separate_likeparams=False, debug=False, **kwargs):
        self.parameters = Map()
        self.debug = debug
        self.separate_likeparams = separate_likeparams
        return

    def init_approx(self, sampler, init_likelihood=True):
        """ Initialize (or reset) ApproxAlgorithm

        Args:
            sampler (GibbsSampler): GibbsSampler

        """
        if not isinstance(sampler, ep_clustering.GibbsSampler):
            raise TypeError("likelihood must be Likelihood object")
        self.K = sampler.K
        if init_likelihood:
            if self.separate_likeparams:
                self.likelihood = [sampler.likelihood.deepcopy()
                        for k in range(self.K)]
                sampler.state.likelihood_parameter = [
                        likelihood.parameter
                        for likelihood in self.likelihood]
            else:
                self.likelihood = sampler.likelihood
        sampler.sample_theta()
        return

    def get_likelihood(self, k=0):
        """ Return likelihood for cluster k """
        if self.separate_likeparams:
            return self.likelihood[k]
        else:
            return self.likelihood

    def loglikelihood_z(self, index, state):
        """ Calculate the loglikelihoods of z[index]

        Args:
            index (int): index of z
            state (Map): map of sampler state

        Returns:
            loglikelihood_z (ndarray): K likelihood values
        """
        raise NotImplementedError()

    def update_approx(self, index, old_z, new_z):
        """ Update the ApproxAlgorithm for new assignment z[index] = new_z """
        return

    def sample_theta(self, state):
        """ Return a sample from the posterior given z, y """
        sampled_theta = np.array([
            self.get_likelihood(k).sample(
                indices = np.where(state.z == k)[0],
                prior_parameter=self.get_likelihood(k).theta_prior,
                )
            for k in range(0, self.K) ])
        return sampled_theta

    def sample_likelihood_parameters(self, state, parameter_name = None):
        if self.separate_likeparams:
            for k in range(self.K):
                self.get_likelihood(k).update_local_parameters(
                    k = k,
                    z = state.z,
                    theta = state.theta,
                    parameter_name = parameter_name)
        else:
            self.get_likelihood().update_parameters(
                z = state.z,
                theta = state.theta,
                parameter_name = parameter_name)
        return

@fix_docs
class NaiveAlgorithm(ApproxAlgorithm):
    name = "naive"
    def loglikelihood_z(self, index, state):
        loglikelihood_z = np.zeros(self.K)
        for k in range(0, self.K):
            loglikelihood_z[k] = \
                    self.get_likelihood(k).loglikelihood(index, state.theta[k])

        return loglikelihood_z

@fix_docs
class CollapsedAlgorithm(ApproxAlgorithm):
    name = "collapsed"
    def loglikelihood_z(self, index, state):
        loglikelihood_z = np.zeros(self.K)
        for k in range(0, self.K):
            subset_indices = _get_cluster(index=index, cluster=k, state=state)
            loglikelihood_z[k] = \
                    self.get_likelihood(k).collapsed(index,
                            subset_indices,
                            self.get_likelihood(k).theta_prior)

        return loglikelihood_z

def _get_cluster(index, cluster, state, max_size=np.inf):
    """ Get Cluster Indices
    Args:
      index (int): observation index
      cluster (int): cluster index
      max_size (int): maximum cluster size (ignoring the observation)
      state (Map): state of GibbsSampler

    Returns:
      subset_indices (max_size ndarray): indices of data sampled
    """
    subset_indices = np.where(state.z == cluster)[0]
    if state.z[index] == cluster:
        subset_indices = subset_indices[subset_indices != index]
    if np.size(subset_indices) > max_size:
        subset_indices = np.sort(np.random.choice(subset_indices,
            size=max_size, replace=False))
    return subset_indices

@fix_docs
class EPAlgorithm(ApproxAlgorithm):
    """ EP Algorithm Class for GibbsSampler

    Args:
        exp_family (ExponentialFamily): exponential family of approx
        damping_factor (double): damped updates (default = 0.0)

    Attributes:
        name (string): name of algorithm
        likelihood (Likelihood): model likelihood
        parameters (Map): approximation parameters
            'site_approx': likelihood site approximations
            'post_approx': posterior approx to theta

    Methods:
        init_approx(sampler)
        loglikelihood_z(index, state): return approx loglikelihoods of z[index]
        update_approx(index, new_z): update approximation

    """
    name = "EP"
    def __init__(self, exp_family, damping_factor=0.0, **kwargs):
        super(EPAlgorithm, self).__init__(**kwargs)

        # exp_family
        if not issubclass(exp_family, ep_clustering.ExponentialFamily):
            raise ValueError("exp_family must be an ExponentialFamily Class")
        self.exp_family = exp_family

        # damping_factor
        if not isinstance(damping_factor, float):
            raise TypeError("damping_factor {0} must be float".format(
                damping_factor))
        if damping_factor < 0.0 or damping_factor >= 1.0:
            raise ValueError("damping_factor {0} must be in [0,1)".format(
                damping_factor))
        self.damping_factor = damping_factor

        return

    def init_approx(self, sampler, init_likelihood=True):
        if not isinstance(sampler, ep_clustering.GibbsSampler):
            raise TypeError("likelihood must be Likelihood object")
        self.K = sampler.K
        if init_likelihood:
            if self.separate_likeparams:
                self.likelihood = [sampler.likelihood.deepcopy()
                        for k in range(self.K)]
                sampler.state.likelihood_parameter = [
                        likelihood.parameter
                        for likelihood in self.likelihood]
            else:
                self.likelihood = sampler.likelihood

        theta_prior = self.get_likelihood().theta_prior
        if not isinstance(theta_prior, self.exp_family):
            raise TypeError("likelihood prior does not match EP exp_family")
        parameters = Map(
                post_approx = [ theta_prior.copy()
                    for k in range(0, self.K) ],
                site_approx = [ self.exp_family(num_dim = sampler.num_dim)
                    for ii in range(0, sampler.num_obs) ],
                )
        self.parameters.update(parameters)
        sampler.sample_theta()
        sampler.update_approx_alg()
        self._sampler = sampler
        return

    def loglikelihood_z(self, index, state):
        loglikelihood_z = np.zeros(self.K)

        z_index = state.z[index]
        cavity_approx = self._calc_cavity_approx(index, z_index)
        try:
            for k in range(0, self.K):
                loglikelihood_z[k] = self.get_likelihood(k).ep_loglikelihood(
                        index=index,
                        theta_parameter=cavity_approx[k],
                        )
            if np.any(np.isnan(loglikelihood_z)):
                raise RuntimeError("nan in loglikelihood_z")
        except:
            logger.warning("EP cavity for site index {0} was invalid".format(index))
            loglikelihood_z = -100*np.ones(self.K)
            loglikelihood_z[z_index] = 0.0

        return loglikelihood_z

    def reset_cluster(self, k, recurse=True):
        """ Reset EP approximation for cluster k """
        if not recurse or self.debug:
            raise RuntimeError("EP approximation is no longer valid")
        theta_prior = self.get_likelihood().theta_prior
        self.parameters.post_approx[k] = theta_prior.copy()

        z = self._sampler.state.z
        cluster_indices = np.where(z == k)[0]
        for ii in cluster_indices:
            self.parameters.site_approx[ii] = self.exp_family(
                    num_dim = self._sampler.num_dim)

        np.random.shuffle(cluster_indices)
        for ii in cluster_indices:
            self.update_approx(index=ii,
                old_z=k, new_z=k)
        return

    def _update_approx(self, index, old_z, new_z):
        # Cavity Distribution
        new_post_for_old_z = (self.parameters.post_approx[old_z] - \
                self.parameters.site_approx[index])
        if not new_post_for_old_z.is_valid_density():
            logger.warning("Invalid Density when updating index {0}".format(
                index))
            self.reset_cluster(old_z, recurse=True)
        else:
            self.parameters.post_approx[old_z] = new_post_for_old_z.normalize()

        # Get moments of the tilted distribution
        unnormalized_post_approx = \
                self.get_likelihood(new_z).moment(
                        index=index,
                        theta_parameter=self.parameters.post_approx[new_z],
                        )

        if self.damping_factor == 0.0:
            # Calculate Site Approx
            self.parameters.site_approx[index] = \
                unnormalized_post_approx - self.parameters.post_approx[new_z]
            # Correct log_scaling_coef of site_approx
            # See bottom of page 6 of "EP for Exp Families" by Seeger 2007
            self.parameters.site_approx[index].log_scaling_coef += \
                self.parameters.post_approx[new_z].logpartition() \
                - unnormalized_post_approx.logpartition()

            if np.isnan(self.parameters.site_approx[index].log_scaling_coef):
                raise ValueError("Log Scaling Coef is Nan -> Something Broke")

            # Set new posterior approximation
            self.parameters.post_approx[new_z] = \
                    unnormalized_post_approx.normalize()

        else:
            # Partial Damped Update
            new_site = \
                unnormalized_post_approx - self.parameters.post_approx[new_z]
            new_site.log_scaling_coef += \
                self.parameters.post_approx[new_z].logpartition() \
                - unnormalized_post_approx.logpartition()
            if np.isnan(new_site.log_scaling_coef):
                raise ValueError("Log Scaling Coef is Nan -> Something Broke")
            self.parameters.site_approx[index] *= self.damping_factor
            new_site *= (1.0-self.damping_factor)
            self.parameters.site_approx[index] += new_site

            # Set new posterior approximation
            self.parameters.post_approx[new_z] = (
                    self.parameters.post_approx[new_z] + \
                    self.parameters.site_approx[index]
                    ).normalize()

        return

    def update_approx(self, index, old_z, new_z):
        if old_z == new_z and not self.debug:
            try:
                self._update_approx(index, old_z, new_z)
            except Exception as error:
                    logger.warning("EP update for site {0} had an error".format(index))
                    logger.warning(error)
        else:
            self._update_approx(index, old_z, new_z)
        return

    def _calc_cavity_approx(self, index, cluster):
        # Helper function for removing (index) from (cluster)'s approximation
        cavity_approx = [None]*self.K
        for k in range(0, self.K):
            if cluster == k:
                cavity_approx[k] = \
                    self.parameters.post_approx[k] - \
                    self.parameters.site_approx[index]
            else:
                cavity_approx[k] = self.parameters.post_approx[k].copy()
        return cavity_approx

    # Need to fix timeseries_likelihood to not need `.sample` theta to update x
    def sample_theta(self, state):
        """ Return a sample from the posterior given z, y """
        try:
            return super(EPAlgorithm, self).sample_theta(state)
        except NotImplementedError:
            sampled_theta = np.array([
                self.parameters.post_approx[k].sample()
                for k in range(0, self.K) ])
            return sampled_theta
        else:
            raise
# EOF
