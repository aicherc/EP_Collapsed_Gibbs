#!/usr/bin/env python
"""

Likelihood Objects for Gibbs Sampler

"""

# Import Modules
import numpy as np

# Author Information
__author__ = "Christopher Aicher"


class KalmanFilter(object):
    """ Kalman Filter Object

    N - dimension of state vector
    T - number of time points

    Args:
        y (N by T ndarray): observations
        y_count (N by T ndarray): counts of observations (0 indicates missing)
        A (N ndarray): AR Coefficients (diagonal matrix)
        lambduh (N ndarray): factor loadings
        sigma2_x (double): variance of latent process
        sigma2_y (N ndarray): variance of observation errors (diagonal matrix)
        eta_mean (T ndarray): latent cluster mean
        eta_var (T ndarray): latent cluster variance
        mu_0 (N ndarray): prior mean for x at time -1
        V_0 (N by N ndarray): prior variance for x at time -1

    Attributes:
        y_T (T by N ndarray): observations
        y_count_T (T by N ndarray): counts of observations (0 indicates missing)
        A (N ndarray): AR Coefficients (diagonal matrix)
        lambduh (N ndarray): factor loadings
        sigma2_x (double): variance of latent process
        sigma2_y (N ndarray): variance of observation errors (diagonal matrix)
        eta_mean (T ndarray): latent cluster mean eta_var (T ndarray): latent cluster variance

    Methods:
        - kalman_filter_step
        - filter_pass
        - smoothing_pass
        - calculate_log_likelihood
        - calculate_cond_log_likelihood
        - sample_x
        - sample_eta
    """
    def __init__(self, y, A, lambduh, sigma2_x, sigma2_y, eta_mean, eta_var,
            mu_0=None, V_0=None, y_count=None):
        if np.isscalar(A):
            A = np.array([A])
        if np.isscalar(lambduh):
            lambduh = np.array([lambduh])

        if np.isscalar(sigma2_y):
            sigma2_y = np.array([sigma2_y])

        self.y_T = y.T
        self.T, self.N = np.shape(self.y_T)
        self.A = A
        self.lambduh = lambduh
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.eta_mean = eta_mean
        self.eta_var = eta_var
        if mu_0 is None:
            self.mu_0 = np.zeros(self.N)
        if V_0 is None:
            self.V_0 = np.ones(self.N)
            self.V_0 *= self.sigma2_x/(1.0-self.A**2)
            self.V_0 = np.diag(self.V_0)
        if y_count is None:
            y_count = 1.0 - np.isnan(y)
        y_count[np.isnan(y)] = 0
        self.y_count_T = y_count.T

        # Scalar Division is much more efficient that np.linalg.solve
        if self.N == 1:
            self.linalg_solve = lambda a, x: x/a
        else:
            self.linalg_solve = np.linalg.solve

        self._check_attrs()
        return

    def _check_attrs(self):
        """ Check that attrs are valid """
        if np.size(self.A) != self.N:
            raise ValueError("A must be a N ndarray")
        if np.size(self.lambduh) != self.N:
            raise ValueError("lambduh must be a N ndarray")
        if np.size(self.sigma2_y) != self.N:
            raise ValueError("sigma2_y must be a N ndarray")
        if np.any(self.sigma2_y < 0):
            raise ValueError("sigma2_y must be nonnegative")
        if self.sigma2_x < 0:
            raise ValueError("sigma2_x must be nonnegative")
        if np.size(self.eta_mean) !=  self.T:
            raise ValueError("eta_mean must be a T ndarray")
        if np.size(self.eta_var) != self.T:
            raise ValueError("eta_var must be a T ndarray")
        if np.any(self.eta_var < 0):
            raise ValueError("eta_var must be nonnegative")
        if np.size(self.mu_0) != self.N:
            raise ValueError("mu_0 must be a N ndarray")
        if np.shape(self.V_0) != (self.N, self.N):
            raise ValueError("V_0 must be a N by N ndarray")
        if np.any(np.linalg.eigvals(self.V_0) < 0):
            raise ValueError("V_0 must be nonnegative")
        if np.shape(self.y_count_T) != np.shape(self.y_T):
            raise ValueError("y_count and y do not have the same shape")
        if np.any(self.y_count_T < 0):
            raise ValueError("y_count must be nonnegative")

        return

    def kalman_filter_step(self, t, mu_prev, V_prev):
        """ Apply Kalman Filter to new observation at time t
        Args:
            t (int): time index t
            mu_prev (N ndarray): filtered mean at time t-1
            V_prev (N by N ndarray): filtered variance at time t-1

        Returns:
            out (dict): dictionary containing
                - mu_filter (N ndarray) - filtered mean at time t
                - V_filter (N by N ndarray) - filtered variance at time t
                - S_t (N by N ndarray) - predictive variance for observation y_t
                - mu_predict (N ndarray) - predictive mean for time t
                - V_predict (N by N ndarray) - predictive variance for time t
        """
        # Predict
        y_t = self.y_T[t]
        y_count_t = self.y_count_T[t]
        mu_predict = self.A * mu_prev + self.lambduh * self.eta_mean[t]
        Q = (np.eye(self.N)*self.sigma2_x +
                np.outer(self.lambduh, self.lambduh)*self.eta_var[t])
        V_predict = _mult_diag_matrix(self.A,
                _mult_diag_matrix(self.A, V_prev, on_right=True)) + Q

        is_obs = y_count_t > 0
        V_yx = V_predict[is_obs,:]
        V_yy = V_yx[:,is_obs]

        if np.any(is_obs):
            # Observation Variance
            S_t = V_yy + np.diag(self.sigma2_y[is_obs] / y_count_t[is_obs])
            if np.any(np.isnan(S_t)):
                raise ValueError("DEBUG")

            # Gain Matrix
            K_t = self.linalg_solve(S_t, V_yx).T

            # Filter
            mu_filter = mu_predict + K_t.dot(y_t[is_obs] - mu_predict[is_obs])
            V_filter =  V_predict - K_t.dot(V_yx)
        else:
            # No observations -> No filter update step
            S_t = np.array([])
            mu_filter = mu_predict
            V_filter = V_predict

        out = {
                'mu_predict': mu_predict,
                'V_predict': V_predict,
                'S_t': S_t,
                'mu_filter': mu_filter,
                'V_filter': V_filter, }
        return out

    def filter_pass(self):
        """ One pass of the Kalman Filter
        Returns:
            out (list of T dicts): containing
                - mu_filter (N ndarray) - filtered mean at time t
                - V_filter (N by N ndarray) - filtered variance at time t
                - S_t (N by N ndarray) - predictive variance for observation y_t
                - mu_predict (N ndarray) - predictive mean for time t
                - V_predict (N by N ndarray) - predictive variance for time t
        """
        mu = self.mu_0
        V = self.V_0
        out = [None]*self.T
        for t in range(0, self.T):
            out[t] = self.kalman_filter_step(t, mu, V)
            mu, V = out[t]['mu_filter'], out[t]['V_filter']
        return out

    def calculate_log_likelihood(self):
        """ Calculate the log-likelihood of y

        Returns:
            log_like (double): log-likelihood of observations
        """
        log_like = 0.0
        mu = self.mu_0
        V = self.V_0
        for t in range(0, self.T):
            kalman_result = self.kalman_filter_step(t, mu, V)
            y_t = self.y_T[t]
            y_count_t = self.y_count_T[t]
            is_obs = y_count_t > 0
            log_like += _gaussian_log_likelihood(y_t[is_obs],
                    mean=kalman_result['mu_predict'][is_obs],
                    variance=kalman_result['S_t'])
            mu, V = kalman_result['mu_filter'], kalman_result['V_filter']
        return np.asscalar(log_like)

    def calculate_cond_log_likelihood(self, i):
        """ Calculate the conditional log-likelihood of y_i given other y

        Args:
            i (int): index of stream

        Returns:
            cond_log_like (double): conditional log-likelihood of stream i
        """
        cond_log_like = 0.0
        mu = self.mu_0
        V = self.V_0
        for t in range(0, self.T):
            kalman_result = self.kalman_filter_step(t, mu, V)
            y_t = self.y_T[t]
            y_count_t = self.y_count_T[t]
            is_obs = y_count_t > 0
            if is_obs[i]:
                cond_log_like += _gaussian_cond_log_likelihood(
                        x=y_t[is_obs],
                        mean=kalman_result['mu_predict'][is_obs],
                        variance=kalman_result['S_t'],
                        i=(np.cumsum(is_obs)[i] - 1),
                        )
            mu, V = kalman_result['mu_filter'], kalman_result['V_filter']
        return cond_log_like

    def smoothing_pass(self, filter_out=None, calc_prev=False):
        """ One pass of the Kalman Smoothing

        Args:
            filter_out (list of dicts): output of filter_pass (optional)
                Will call `filter_pass` if not supplied
            calc_prev (bool): calculate smoothed posterior for t=-1

        Returns:
            out (list of T dicts): containing
                - mu_smoothed (N ndarray) - filtered mean at time t
                - V_smoothed (N by N ndarray) - filtered variance at time t
                - J_t (N by N ndarray) - backward filter matrix

            If calc_prev is True, then smoothing_pass() will also return
                the dict prev (for t=-1)
        """
        out = [None]*self.T


        # Forward Kalman Filter
        if filter_out is None:
            filter_out = self.filter_pass()

        # Backward Smoothing Pass
        mu_smoothed = filter_out[self.T-1]['mu_filter']
        V_smoothed = filter_out[self.T-1]['V_filter']
        out[self.T-1] = {'mu_smoothed': mu_smoothed,
                         'V_smoothed': V_smoothed,
                         'J_t': None}
        for t in reversed(range(0, self.T-1)):
            mu_filter = filter_out[t]['mu_filter']
            V_filter = filter_out[t]['V_filter']
            mu_predict_next = filter_out[t+1]['mu_predict']
            V_predict_next = filter_out[t+1]['V_predict']

            J_t = self.linalg_solve(V_predict_next,
                    _mult_diag_matrix(self.A, V_filter)).T

            mu_smoothed = mu_filter + J_t.dot(mu_smoothed-mu_predict_next)
            V_smoothed = (V_filter +
                    J_t.dot(V_smoothed - V_predict_next).dot(J_t.T))

            out[t] = {'mu_smoothed': mu_smoothed,
                      'V_smoothed': V_smoothed,
                      'J_t': J_t}
        if not calc_prev:
            return out

        else:
            # Handle t = -1
            mu_filter = self.mu_0
            V_filter = self.V_0
            mu_predict_next = filter_out[0]['mu_predict']
            V_predict_next = filter_out[0]['V_predict']

            J_t = self.linalg_solve(V_predict_next,
                    _mult_diag_matrix(self.A, V_filter)).T

            mu_smoothed = mu_filter + J_t.dot(mu_smoothed-mu_predict_next)
            V_smoothed = (V_filter +
                    J_t.dot(V_smoothed - V_predict_next).dot(J_t.T))

            prev = {'mu_smoothed': mu_smoothed,
                    'V_smoothed': V_smoothed,
                    'J_t': J_t}
            return out, prev

    def _backward_pass(self, filter_out = None, smoothing_out = None):
        """ Helper function for moments of G(X_t) ~ Pr(Y_{t:T} | X_t)

        G(X_t) ~ Pr(X_t | Y_{1:T}) / Pr(X_t | Y_{1:t-1})

        Returns:
            out (list of T dicts): containing
                - mu_beta (N ndarray) - backward filtered mean at time t
                - V_beta (N by N ndarray) - backward filtered variance at time t
        """
        out = [None]*self.T

        # Perform Filter and Smoother if necessary
        if filter_out is None:
            filter_out = self.filter_pass()

        if smoothing_out is None:
            smoothing_out = self.smoothing_pass(filter_out = filter_out)

        for t in range(0, self.T):
            mu_predict = filter_out[t]['mu_predict']
            V_predict = filter_out[t]['V_predict']
            mu_smoothed = smoothing_out[t]['mu_smoothed']
            V_smoothed = smoothing_out[t]['V_smoothed']

            if np.allclose(V_smoothed, V_predict):
                # If Pr(Y_{s:T} | X_s) = 1, e.g. no observations in s:T
                # Then set V_beta = Inf
                V_beta = np.diag(np.inf * np.ones(self.N))
                mu_beta = np.zeros(self.N)

            else:
                V_beta = V_smoothed.dot(
                        np.eye(self.N) +
                        self.linalg_solve(V_predict - V_smoothed, V_smoothed)
                        )
                mu_beta = V_beta.dot(
                        self.linalg_solve(V_smoothed, mu_smoothed) -
                        self.linalg_solve(V_predict, mu_predict)
                        )

            out[t] = {
                    "mu_beta": mu_beta,
                    "V_beta": V_beta,
                    }

        return out


    def moment_eta(self):
        """ Return the mean and (diag) variance of the latent process given Y.

        Returns the marginal moments of likelihood fo the latent process for EP.

        Note that eta_mean, eta_variance are the parameters of [Pr(Y | \eta_s)]

        Returns:
            eta_mean (T ndarray): mean of eta likelihood
            eta_variance (T ndarray): variance of eta likelihood
        """
        eta_mean = np.zeros(self.T)
        eta_variance = np.zeros(self.T)

        filter_out = self.filter_pass()
        smoothing_out = self.smoothing_pass(filter_out = filter_out)

        beta_out = self._backward_pass(
                filter_out = filter_out,
                smoothing_out = smoothing_out
                )

        # Constants
        sigma2_eta = (self.lambduh.dot(self.lambduh))**-1 * self.sigma2_x
        p_beta = (self.lambduh.dot(self.lambduh))**-1 * self.lambduh
        p_alpha = -1.0 * p_beta * self.A

        for t in range(0, self.T):
            # alpha(X_{t-1}) ~ Pr(X_{t-1} | Y_{1:t-1})
            if t == 0:
                mu_alpha = self.mu_0
                V_alpha = self.V_0
            else:
                mu_alpha = filter_out[t-1]["mu_filter"]
                V_alpha = filter_out[t-1]["V_filter"]

            # beta(X_t) ~ Pr(Y_{t:T} | X_t)
            mu_beta = beta_out[t]["mu_beta"]
            V_beta = beta_out[t]["V_beta"]

            eta_mean[t] = p_alpha.dot(mu_alpha) + p_beta.dot(mu_beta)
            eta_variance[t] = (
                    p_alpha.dot(V_alpha.dot(p_alpha)) +
                    p_beta.dot(V_beta.dot(p_beta)) +
                    sigma2_eta
                    )

        return eta_mean, eta_variance


    def _old_moment_eta(self):
        """ Old (incorrect) EP moment update step

        Use `moment_eta` instead.

        Return the mean and variance of the likelihood of the
        latent process given Y (integrating out X).

        Returns:
            eta_mean (T ndarray): mean of eta
            eta_variance (T ndarray): variance of eta

        """
        eta_mean = np.zeros(self.T)
        eta_variance = np.zeros(self.T)

        smoothing_out, prev = self.smoothing_pass(calc_prev=True)

        # Handle t = 0
        J_prev = prev['J_t']
        mu_prev = prev['mu_smoothed']
        V_prev = prev['V_smoothed']
        mu = smoothing_out[0]['mu_smoothed']
        V = smoothing_out[0]['V_smoothed']

        eta_mean[0] = (mu - self.A * mu_prev) / self.lambduh
        eta_variance[0] = (self.sigma2_x +
                    (V + self.A**2 * V_prev - 2 * V * J_prev * self.A) /
                    (self.lambduh**2))

        # Handle t = 1:T-1
        for t in range(1, self.T):
            J_prev = smoothing_out[t-1]['J_t']
            mu_prev = mu
            V_prev = V
            mu = smoothing_out[t]['mu_smoothed']
            V = smoothing_out[t]['V_smoothed']

            eta_mean[t] = (mu - self.A * mu_prev) / self.lambduh
            eta_variance[t] = (self.sigma2_x +
                    (V + self.A**2 * V_prev - 2 * V * J_prev * self.A) /
                    (self.lambduh**2))

        return eta_mean, eta_variance

    def sample_x(self, filter_out=None):
        """ Sample latent process using forward filter backward sampler

        Args:
            filter_out (list of dicts): output of filter_pass (optional)
                Will call filter_pass if not supplied

        Returns:
            x (T by N ndarray): sample from latent state conditioned on y
        """
        x = np.zeros((self.T,self.N))

        # Forward Kalman Filter
        if filter_out is None:
            filter_out = self.filter_pass()

        # Backwards Sampler
        mu = filter_out[self.T-1]['mu_filter']
        V = filter_out[self.T-1]['V_filter']
        #x_next = np.random.multivariate_normal(mean=mu, cov=V)
        x_next = _sample_multivariate_normal(mu, V)
        x[self.T-1,:] = x_next
        for t in reversed(range(0, self.T-1)):
            mu_filter = filter_out[t]['mu_filter']
            V_filter = filter_out[t]['V_filter']
            mu_predict_next = filter_out[t+1]['mu_predict']
            V_predict_next = filter_out[t+1]['V_predict']

            J_t = self.linalg_solve(V_predict_next,
                    _mult_diag_matrix(self.A, V_filter)).T
            mu = mu_filter + J_t.dot(x_next - mu_predict_next)
            V = V_filter - J_t.dot(_mult_diag_matrix(self.A, V_filter))
            # x_next = np.random.multivariate_normal(mu, V)
            x_next = _sample_multivariate_normal(mu, V)

            x[t,:] = x_next
        return x


    def sample_eta(self, x=None):
        """ Sample latent process

        Args:
            x (T by N ndarray): sampled x (optional)

        Returns:
            eta (T ndarray): sampled eta
        """
        if x is None:
            x = self.sample_x()
        eta = np.zeros(self.T)

        # Handle t = 0
        mean_1 = self.eta_mean[0]
        var_1 = self.eta_var[0]
        mean_2 = np.sum(
                self.lambduh * (x[0] - self.A * self.mu_0)
                ) / np.sum(self.lambduh ** 2)
        var_2 = np.sum(
                self.lambduh ** 2 /
                (self.sigma2_x + self.A**2 * np.diag(self.V_0))
                ) ** -1
        var = 1.0/(1.0/var_1 + 1.0/var_2)
        mean = (mean_1/var_1 + mean_2/var_2) * var
        eta[0] = np.random.randn(1)*np.sqrt(var) + mean

        # Handle t = 1:T-1
        for t in range(1, self.T):
            mean_1 = self.eta_mean[t]
            var_1 = self.eta_var[t]

            mean_2 = np.sum(
                    self.lambduh * (x[t] - self.A * x[t-1])
                    ) / np.sum(self.lambduh ** 2)
            var_2 = self.sigma2_x / np.sum(self.lambduh ** 2)

            var = 1.0/(1.0/var_1 + 1.0/var_2)
            mean = (mean_1/var_1 + mean_2/var_2) * var
            eta[t] = np.random.randn(1)*np.sqrt(var) + mean

        return eta

    def sample_y(self, x=None, filter_out=None):
        """ Sample new observations based on latent process conditioned on y

        Args:
            x (T by N ndarray): sample from latent state conditioned on y
            filter_out (list of dicts): output of filter_pass (optional)
                Only used if x is not supplied

        Returns:
            y (T by N ndarray): sample of observations conditioned on y
        """
        y = np.zeros((self.T, self.N))

        # Draw X is not supplied
        if x is None:
            x = self.sample_x(filter_out=filter_out)

        # Y is a noisy version of X
        y = x + _mult_diag_matrix(self.sigma2_y,
            np.random.normal(size=np.shape(x)),
            on_right = True)

        return y


#UTILITY FUNCTION
def _mult_diag_matrix(D, mtx, on_right=False):
    """ Multiply diagonal matrix D to mtx
    Args:
      D (N ndarray) - diagonal matrix
      mtx (ndarray) - matrix to multiply
      on_right (bool) - whether to return D * mtx (False) or mtx * D (True)
    """
    if not on_right:
        return (D*mtx.T).T
    else:
        return D*mtx

def _sample_multivariate_normal(mean, cov):
    """ Alternative to numpy.random.multivariate_normal """
    if np.size(mean) == 1:
        x = np.random.normal(loc = mean, scale = np.sqrt(cov))
        return x
    else:
        L = np.linalg.cholesky(cov)
        x = L.dot(np.random.normal(size = np.size(mean))) + mean
        return x

def _gaussian_log_likelihood(x, mean, variance):
    """ Calculate the log-likelihood of multivariate Gaussian """
    N = np.size(x)
    log_like = - N/2.0 * np.log(2*np.pi)
    if N == 1:
        log_like += - 0.5 * np.log(variance)
        log_like += - 0.5 * (x-mean)**2/variance
    elif N == 0:
        log_like = 0.0
    else:
        log_like += - 0.5 * np.linalg.slogdet(variance)[1]
        log_like += - 0.5 * np.sum((x-mean)*np.linalg.solve(variance, x-mean))
    return log_like

def _gaussian_cond_log_likelihood(x, mean, variance, i):
    """ Calculate the conditional log-likelihood of multivariate Gaussian """
    N = np.size(x)
    if i >= N:
        raise ValueError("Index i is too large for x")
    if N == 1:
        return _gaussian_log_likelihood(x, mean, variance)
    j = np.arange(N) != i
    V_ii = variance[i,i]
    V_ij = variance[i,j]
    V_jj = variance[np.ix_(j,j)]
    mu_i = mean[i]
    mu_j = mean[j]
    K_ij = np.linalg.solve(V_jj, V_ij.T).T
    cond_mean = mean[i] + K_ij.dot(x[j] - mu_j)
    cond_variance = V_ii - K_ij.dot(V_ij.T)
    cond_log_like = _gaussian_log_likelihood(x[i], cond_mean, cond_variance)
    return cond_log_like

def _categorical_sample(probs):
    """ Draw a categorical random variable over {0,...,K-1}
    Args:
      probs (K ndarray) - probability of each value
    Returns:
      draw (int) - random outcome
    """
    return int(np.sum(np.random.rand(1) > np.cumsum(probs)))




#EOF
