#!/usr/bin/env python
"""

Gibbs Sampler

"""

# Import Modules
import numpy as np
import pandas as pd
import logging
from ._utils import Map
from copy import deepcopy
from tqdm import tqdm

# Author Information __author__ = "Christopher Aicher"
logger = logging.getLogger(name=__name__)

# Code Implementation
def random_init_z(N, K, z_prior = None):
    """ Draw Z for N assignments, K cluster/groups """
    if z_prior is None:
        z_prior = np.ones(K)/K
    z = np.random.multinomial(
            n=1,
            pvals=z_prior/np.sum(z_prior),
            size=N,
            ).dot(np.arange(0, K))
    return z

class GibbsSampler(object):
    """ GibbsSampler

    N observations, T dimensions

    Args:
        data (GibbsData): data
        likelihood (Likelihood or string): likelihood object with functions for
        K (int): number of latent clusters
        approx_alg (ApproxAlgorithm or string)
        **kwargs (kwargs): keyword arguments for options
            alg_type (string): algorithm to use when sampling z
                "naive"         : Naive Gibbs (default)
                "collapsed"     : Collapsed Gibbs
                "EP"            : EP Gibbs
                "subsampled"    : Subsampled Collapsed Gibbs
                "EP_subsampled" : EP Gibbs w/Subsampling
            max_subsample_size (int): max subsample size (default: np.inf)
            theta_prior (NaturalParameters): theta prior (default: std norm)
            theta_prior_variance (double): prior variance (default: 100)
            z_prior_type (string): whether to use a fixed prior over z ("fixed")
                or a dirichlet prior ("dirichlet") (default "fixed")
            z_prior (K ndarray): dirichlet prior for Pr(z) (default: 1/K)
            track_ep (bool): whether to compute EP likelihood approximations
                (default True for EP, False otherwise)
            full_mcmc (bool): whether to sample all parameters (default True)
            shuffle (bool): whether to shuffle scan order over z (default True)

    Attributes:
        num_obs (int): number of observations
        num_dim (int): number of dimensions
        likelihood(Likelihood): likelihood object
        K (int): number of latent clusters
        approx_alg (ApproxAlgorithm)
        state (Map): state of sampler consists of:
            z (N ndarray): latent cluster assignment
            theta (K by T ndarray): cluster parameters
            likelihood_parameters (Map): map of likelihood parameters
            approx_parameters (Map): map of approx algorithm parameters
        options (Map): algorithm options

    Methods:
        - init_state() -> randomly select initial state
        - one_step() -> one step of GibbSampler
        - sample_z() -> sample z
        - sample_theta() -> sample theta condition on z
        - eval_loglikelihood() -> return loglikelihood
        - sample_likelihood_parameters() -> sample likelihood parameters
        - get_state() -> return dictionary with current state
        - set_state() -> set current state with a dictionary
    """
    def __init__(self, data, likelihood, K, approx_alg, **kwargs):
        self.num_obs = data.num_obs
        self.num_dim = data.num_dim
        self.K = K
        self.likelihood = likelihood
        self.approx_alg = approx_alg
        self.options = self._process_options(**kwargs)
        self._set_prior()
        self.init_state(init_z = self.options.init_z)
        return

    @staticmethod
    def _process_options(**kwargs):
        # Default Options
        options = {
                "max_subsample_size": np.inf,
                "theta_prior": None,
                "theta_prior_variance": 1.0,
                "z_prior": None,
                "z_prior_type": "fixed",
                "init_z": None,
                "full_mcmc": True,
                "shuffle": True,
                "verbosity": logging.INFO,
                "log_file": None,
                }

        # Parse Input
        for key, value in kwargs.items():
            if key in options.keys():
                options[key] = value
            else:
                print("Ignoring unrecognized kwarg: {0}".format(key))
        return Map(options)

    def _set_prior(self):
        # Set Default Priors
        if self.options.z_prior is None:
            self.options.z_prior = np.ones(self.K)/self.K
        elif np.size(self.options.z_prior) == 1:
            self.options.z_prior = np.ones(self.K) * self.options.z_prior
        return

    def init_state(self, init_z = None):
        """ Initialize State """
        self.state = Map()

        if init_z is None:
            self.state['z'] = \
                    random_init_z(self.num_obs, self.K, self.options.z_prior)
        else:
            self.state['z'] = np.copy(init_z)

        self.state['theta'] = np.zeros((self.K, self.num_dim))
        self.state['likelihood_parameter'] = self.likelihood.parameter
        self.state['approx_parameters'] = self.approx_alg.parameters
        self.approx_alg.init_approx(sampler=self)
        return

    def get_state(self):
        """ Return a deepcopy of state """
        state = Map(deepcopy(self.state))
        return state

    def save_state(self, filename, filetype=None):
        """ Save state to file (see Map.save) """
        self.state.save(filename, filetype)
        return

    def load_state(self, filename, filetype=None):
        """ Load state from file (see Map.load) """
        self.state.load(filename, filetype)
        self.likelihood.parameter = self.state.likelihood_parameter
        self.approx_alg.parameters = self.state.approx_parameters
        return

    def set_state(self, **kwargs):
        """ Set state """
        for key, value in kwargs.items():
            if key in self.state.keys():
                if isinstance(value, Map):
                    self.state[key].update(deepcopy(value))
                else:
                    self.state[key] = deepcopy(value)
            else:
                logger.warning("Unrecognized key '%s' for state", key)

        #self._check_state()
        return

    def one_step(self, full_mcmc=None):
        self.sample_z()
        self.sample_theta()
        if full_mcmc is None:
            full_mcmc = self.options.full_mcmc
        if full_mcmc:
            self.sample_likelihood_parameters()

        return

    def sample_theta(self):
        """ Return a sample from posterior given z,y """


        self.state.theta = self.approx_alg.sample_theta(self.state)
        return self.state.theta

    def _create_scan_order_gen(self):
        """ Generator for the next observation """
        while(True):
            scan_order = range(0, self.num_obs)
            if self.options.shuffle:
                np.random.shuffle(list(scan_order))
            for ii in scan_order:
                yield ii

    def sample_one_z(self, ii=None):
        """ Update a single observation's cluster assignment z_i

        Args:
            ii (int): index of observation to sample cluster assignments
                If not specified, will update in scan_order
        Returns:
            out (tuple): [ii, old_z_i, new_z_i]
        """
        if ii is None:
            if not hasattr(self, "_scan_order_gen"):
                self._scan_order_gen = self._create_scan_order_gen()
            ii = next(self._scan_order_gen)
        logging.debug("Sampling z[%u]", ii)
        # Calculate Log Posterior
        if self.options.z_prior_type == "fixed":
            logprior = np.log(self.options.z_prior)
        elif self.options.z_prior_type == "dirichlet":
            dir_post_alpha = np.bincount(self.state.z) + self.options.z_prior
            logprior = np.log(dir_post_alpha) - np.log(np.sum(dir_post_alpha))
        else:
            raise ValueError("z_prior_type unrecognized {0}".format(
                self.options.z_prior_type))
        loglikelihood = \
                self.approx_alg.loglikelihood_z(index=ii, state=self.state)

        logposterior = logprior+loglikelihood
        logposterior -= np.max(logposterior)
        posterior = np.exp(logposterior)
        posterior = posterior / np.sum(posterior)

        # Sample new_z_i (int)
        new_z_i = np.random.multinomial(1,posterior,1).dot(
                np.arange(0, self.K))[0]

        # Update approx + state
        old_z_i = self.state.z[ii]
        self.approx_alg.update_approx(index=ii,
                old_z=old_z_i, new_z=new_z_i)
        self.state.z[ii] = new_z_i

        out = [ii, old_z_i, new_z_i]
        return out

    def sample_z(self):
        """ Return a sample of z from the posterior

        Also updates approx_alg based on new samples

        Returns:
          sampled_z (N ndarray): latent cluster assignments
        """
        scan_order = range(0, self.num_obs)
        if self.options.shuffle:
            np.random.shuffle(list(scan_order))
        for ii in scan_order:
            self.sample_one_z(ii)
        return self.state.z

    def update_approx_alg(self, scan_order=None):
        """ Updates approx_alg """
        logger.info("Updating Approx Alg")
        if scan_order is None:
            scan_order = range(0, self.num_obs)
        if self.options.shuffle:
            np.random.shuffle(list(scan_order))
        for ii in scan_order:
            self.approx_alg.update_approx(
                        index=ii,
                        old_z=self.state.z[ii],
                        new_z=self.state.z[ii])
        return

    def reset_approx_alg(self):
        """ Reset approx_alg """
        logger.info("Resetting Approx Alg")
        self.approx_alg.init_approx(self, init_likelihood=False)
        return

    def sample_likelihood_parameters(self, parameter_name = None):
        likeparams = self.approx_alg.sample_likelihood_parameters(
                self.state, parameter_name=parameter_name)
        return likeparams

    def eval_loglikelihood(self, kind="naive"):
        """ Return the loglikelihood of the current state

        Args:
            kind (string):
                'naive': loglikelihood(y | z, theta)
                'collapsed': loglikelihood(y | z)
        """
        # TODO: ADD 'alg_estimate': alg's estimate for loglikelihood(y | z)
        loglikelihood = 0.0

        if kind == "naive":
            for ii in range(0, self.num_obs):
                z_ii = self.state.z[ii]
                likelihood = self.approx_alg.get_likelihood(z_ii)
                loglikelihood += likelihood.loglikelihood(
                        index=ii, theta=self.state.theta[z_ii])

        elif kind == "collapsed":
            for cluster in range(0, self.K):
                cluster_indices = np.where(self.state.z == cluster)[0]
                likelihood = self.approx_alg.get_likelihood(cluster)
                loglikelihood += likelihood.cluster_loglikelihood(
                        indices=cluster_indices,
                        theta_parameter=self.options.theta_prior)

        else:
            raise ValueError("Unrecognized kind={0}".format(kind))

        return loglikelihood


# Code to execute if called from command-line
if __name__ == '__main__':
    print("gibbs")





# EOF
