# distutils: language = c++
# distutils: sources = cython/src/c_kalman_filter.cpp

# ^ distutils lines are important

# Import Eigen types (e.g. VectorXd, MatrixXd, numpy_copy, np.ndarray)
from eigency.core cimport *
from libcpp.vector cimport vector
from ep_clustering.kalman_filter._kalman_filter import _mult_diag_matrix
import numpy as np
import logging


# Pythonic redefinition of Kalman Filter class header for Cython
cdef extern from "../cython/src/c_kalman_filter.h" namespace "c_kalman_filter":
    cdef cppclass KalmanFilterStepValues:
        KalmanFilterStepValues() except+
        KalmanFilterStepValues(
            Map[VectorXd] &mean_predict,
            Map[MatrixXd] &variance_predict,
            Map[MatrixXd] &variance_observation,
            Map[VectorXd] &mean_filter,
            Map[MatrixXd] &variance_filter
            ) except+
        # Attributes
        VectorXd mean_predict
        MatrixXd variance_predict
        MatrixXd variance_observation
        VectorXd mean_filter
        MatrixXd variance_filter


    cdef cppclass KalmanFilter:
        KalmanFilter() except+
        KalmanFilter(
                Map[MatrixXd] &y_mean,
                Map[MatrixXd] &y_count,
                Map[VectorXd] &A,
                Map[VectorXd] &lambduh,
                Map[VectorXd] &sigma2_y,
                double &sigma2_x,
                Map[VectorXd] &eta_mean,
                Map[VectorXd] &eta_var,
                Map[VectorXd] &mean_x_prior,
                Map[MatrixXd] &variance_x_prior
                ) except+

        # Attribute Get/Set
        MatrixXd get_y_mean()
        void set_y_mean(Map[MatrixXd])
        MatrixXd get_y_count()
        void set_y_count(Map[MatrixXd])
        VectorXd get_A()
        void set_A(Map[VectorXd])
        VectorXd get_lambduh()
        void set_lambduh(Map[VectorXd])
        VectorXd get_sigma2_y()
        void set_sigma2_y(Map[VectorXd])
        double get_sigma2_x()
        void set_sigma2_x(double)
        VectorXd get_eta_mean()
        void set_eta_mean(Map[VectorXd])
        VectorXd get_eta_var()
        void set_eta_var(Map[VectorXd])
        VectorXd get_mean_x_prior()
        void set_mean_x_prior(Map[VectorXd])
        MatrixXd get_variance_x_prior()
        void set_variance_x_prior(Map[MatrixXd])

        # Methods
        KalmanFilterStepValues kalman_filter_step(
                int &t,
                Map[VectorXd] &mean_prev,
                Map[VectorXd] &variance_prev)
        vector[KalmanFilterStepValues] filter_pass()
        double calculate_log_likelihood()
        double calculate_cond_log_likelihood(int &i)
        MatrixXd sample_x()
        VectorXd sample_eta(Map[MatrixXd] &)


# Python Wrappers around the classes
cdef class CKalmanFilter:
    """ C++ Implementation of Kalman Filter

    See `kalman_filter.KalmanFilter` for the python implementation.

    Args:
        y (N by T ndarray): mean observations
        y_count (N by T ndarray, optional): counts of observations
            (0 indicates missing)
        A (N ndarray): AR Coefficients (diagonal matrix)
        lambduh (N ndarray): factor loadings
        sigma2_x (double): variance of latent process
        sigma2_y (N ndarray): variance of observation errors (diagonal matrix)
        eta_mean (T ndarray): latent cluster mean
        eta_var (T ndarray): latent cluster variance
        mu_0 (N ndarray, optional): prior mean for x at time -1
        V_0 (N by N ndarray, optional): prior variance for x at time -1

    Attributes:
        N (int): number of series
        T (int): number of time points
        y (T by N ndarray): observations
        y_count (N by T ndarray): counts of observations (0 indicates missing)
        A (N ndarray): AR Coefficients (diagonal matrix)
        lambduh (N ndarray): factor loadings
        sigma2_x (double): variance of latent process
        sigma2_y (N ndarray): variance of observation errors (diagonal matrix)
        eta_mean (T ndarray): latent cluster mean eta_var (T ndarray): latent cluster variance


    Methods:
        - calculate_log_likelihood
        - calculate_cond_log_likelihood
        - sample_x
        - sample_eta
    """
    cdef KalmanFilter _kalman_filter
    cdef int N
    cdef int T

    def __init__(self,
            y,
            A,
            lambduh,
            sigma2_x,
            sigma2_y,
            eta_mean,
            eta_var,
            y_count = None,
            mu_0 = None,
            V_0 = None):

        y_mean = y # Renaming
        self.N, self.T= np.shape(y_mean)

        # Format Inputs
        if np.isscalar(A):
            A = np.array([A])
        if np.isscalar(lambduh):
            lambduh = np.array([lambduh])
        if np.isscalar(sigma2_y):
            sigma2_y = np.array([sigma2_y])
        if np.ndim(sigma2_x) == 0:
            sigma2_x = np.float64(sigma2_x)
        if y_count is None:
            y_count = 1.0 - np.isnan(y_mean)
        if mu_0 is None:
            mu_0 = np.zeros(self.N)
        if V_0 is None:
            V_0 = np.ones(self.N)
            V_0 *= sigma2_x/(1.0-A**2)
            V_0 = np.diag(V_0)

        y_mean = np.asfortranarray(y_mean)
        y_count = np.asfortranarray(y_count)
        V_0 = np.asfortranarray(V_0)

        self._check_attrs(y_mean, y_count, A, lambduh, sigma2_x, sigma2_y,
            eta_mean, eta_var, mu_0, V_0)

        self._kalman_filter = KalmanFilter(
            Map[MatrixXd](y_mean),
            Map[MatrixXd](y_count),
            Map[VectorXd](A),
            Map[VectorXd](lambduh),
            Map[VectorXd](sigma2_y),
            <double> sigma2_x,
            Map[VectorXd](eta_mean),
            Map[VectorXd](eta_var),
            Map[VectorXd](mu_0),
            Map[MatrixXd](V_0))

        return

    def kalman_filter_step(self, **kwargs):
        raise NotImplementedError()
    def filter_pass(self, **kwargs):
        raise NotImplementedError()
    def smoothing_pass(self, **kwargs):
        raise NotImplementedError()

    def calculate_log_likelihood(self):
        """ Calculate the log-likelihood of y

        Returns:
            log_like (double): log-likelihood of observations
        """
        log_like = self._kalman_filter.calculate_log_likelihood()
        return log_like

    def calculate_cond_log_likelihood(self, i = 0):
        """ Calculate the conditional log-likelihood of y_i given other y
        Args:
            i (int): index of stream

        Returns:
            cond_log_like (double): conditional log-likelihood of stream i
        """
        i = int(i)
        if (type(i) is not int) or (i >= self.N) or (i < 0):
            raise ValueError("i must be an integer in [0, N)")
        cond_log_like = \
                self._kalman_filter.calculate_cond_log_likelihood(<int>i)
        return cond_log_like

    def moment_eta(self):
        """ Return the mean and (diag) variance of the latent process given Y.

        Returns the marginal moments of likelihood fo the latent process for EP.

        Note that eta_mean, eta_variance are the parameters of [Pr(Y | \eta_s)]

        Returns:
            eta_mean (T ndarray): mean of eta likelihood
            eta_variance (T ndarray): variance of eta likelihood
        """
        raise NotImplementedError()


    def sample_x(self, filter_out=None):
        """ Sample latent process using forward filter backward sampler

        Returns:
            x (T by N ndarray): sample from latent state conditioned on y
        """
        if filter_out is not None:
            logging.warning("filter_out is ignored in our cpp implementation")
        x = ndarray_copy(self._kalman_filter.sample_x()).T
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

        if((np.ndim(x) != 2) or
            (np.shape(x)[0] != self.T) or
            (np.shape(x)[1] != self.N)):
            raise ValueError("x must be a T by N ndarray")

        eta = ndarray_copy(self._kalman_filter.sample_eta(
            Map[MatrixXd](np.copy(x.T)))).T[0]
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

    # Define Properties via getter/setters
    property N:
        def __get__(self):
            return self.N
    property T:
        def __get__(self):
            return self.T

    property y:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_y_mean())
        def __set__(self, value):
            if ((np.ndim(value) !=2) or
                (np.shape(value)[0] != self.N) or
                (np.shape(value)[1] != self.T)):
                raise ValueError("y_mean must be N by T array")
            self._kalman_filter.set_y_mean(
                    Map[MatrixXd](np.asfortranarray(value)))

    property y_mean:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_y_mean())
        def __set__(self, value):
            if ((np.ndim(value) !=2) or
                (np.shape(value)[0] != self.N) or
                (np.shape(value)[1] != self.T)):
                raise ValueError("y_mean must be N by T array")
            self._kalman_filter.set_y_mean(
                    Map[MatrixXd](np.asfortranarray(value)))

    property y_count:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_y_count())
        def __set__(self, value):
            if ((np.ndim(value) != 2) or
                (np.shape(value)[0] != self.N) or
                (np.shape(value)[1] != self.T)):
                raise ValueError("y_count must be N by T array")
            self._kalman_filter.set_y_count(
                    Map[MatrixXd](np.asfortranarray(value)))

    property A:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_A()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.N) or (np.ndim(value) != 1):
                raise ValueError("A must be N array")
            self._kalman_filter.set_A(Map[VectorXd](value))

    property lambduh:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_lambduh()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.N) or (np.ndim(value) != 1):
                raise ValueError("lambduh must be N array")
            self._kalman_filter.set_lambduh(Map[VectorXd](value))

    property sigma2_y:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_sigma2_y()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.N) or (np.ndim(value) != 1):
                raise ValueError("sigma2_y must be N array")
            if (np.any(value <= 0)):
                raise ValueError("sigma2_y must be positive")
            self._kalman_filter.set_sigma2_y(Map[VectorXd](value))

    property sigma2_x:
        def __get__(self):
            return np.copy(self._kalman_filter.get_sigma2_x())
        def __set__(self, value):
            if not np.isscalar(value):
                raise TypeError("sigma2_x must be scalar")
            if value <= 0:
                raise ValueError("sigma2_x must be positive")
            self._kalman_filter.set_sigma2_x(<double> value)

    property eta_mean:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_eta_mean()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.T) or (np.ndim(value) != 1):
                raise ValueError("eta_mean must be T array")
            self._kalman_filter.set_eta_mean(Map[VectorXd](value))

    property eta_var:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_eta_var()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.T) or (np.ndim(value) != 1):
                raise ValueError("eta_var must be T array")
            if (np.any(value < 0)):
                raise ValueError("eta_var must be nonnegative")
            self._kalman_filter.set_eta_var(Map[VectorXd](value))

    property mu_0:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_mean_x_prior()).T[0]
        def __set__(self, value):
            if (np.shape(value)[0] != self.N) or (np.ndim(value) != 1):
                raise ValueError("mu_0 must be N array")
            self._kalman_filter.set_mean_x_prior(Map[VectorXd](value))

    property V_0:
        def __get__(self):
            return ndarray_copy(self._kalman_filter.get_variance_x_prior())
        def __set__(self, value):
            if ((np.shape(value)[0] != self.N) or
                (np.ndim(value) != 2) or
                (np.shape(value)[1] != self.N)):
                raise ValueError("V_0 must be N by N array")
            if (np.any(np.linalg.eigvals(value) < 0)):
                raise ValueError("V_0 must be nonnegative definite")
            self._kalman_filter.set_variance_x_prior(
                    Map[MatrixXd](np.asfortranarray(value)))


    def __repr__(self):
        rep = "CKalmanFilter:\n"
        rep += "y_mean:\n" + str(self.y_mean) + "\n"
        rep += "y_count:\n" + str(self.y_count) + "\n"
        rep += "A:\n" + str(self.A) + "\n"
        rep += "lambduh:\n" + str(self.lambduh) + "\n"
        rep += "sigma2_y:\n" + str(self.sigma2_y) + "\n"
        rep += "sigma2_x:\n" + str(self.sigma2_x) + "\n"
        rep += "eta_mean:\n" + str(self.eta_mean) + "\n"
        rep += "eta_var:\n" + str(self.eta_var) + "\n"
        rep += "mu_0:\n" + str(self.mu_0) + "\n"
        rep += "V_0:\n" + str(self.V_0) + "\n"
        return rep

    def _check_attrs(self, y_mean, y_count, A, lambduh, sigma2_x, sigma2_y,
            eta_mean, eta_var, mu_0, V_0):
        """ Check that attrs are valid """
        if ((np.ndim(y_mean) != 2) or
            (np.shape(y_mean)[0] != self.N) or
            (np.shape(y_mean)[1] != self.T)):
            raise ValueError("y_mean must be N by T array")
        if ((np.ndim(y_count) != 2) or
            (np.shape(y_count)[0] != self.N) or
            (np.shape(y_count)[1] != self.T)):
            raise ValueError("y_count must be N by T array")
        if np.size(A) != self.N:
            raise ValueError("A must be a N ndarray")
        if np.size(lambduh) != self.N:
            raise ValueError("lambduh must be a N ndarray")
        if np.size(sigma2_y) != self.N:
            raise ValueError("sigma2_y must be a N ndarray")
        if np.any(sigma2_y <= 0):
            raise ValueError("sigma2_y must be positive")
        if not np.isscalar(sigma2_x):
            raise TypeError("sigma2_x must be a scalar")
        if sigma2_x <= 0:
            raise ValueError("sigma2_x must be positive")
        if np.size(eta_mean) !=  self.T:
            raise ValueError("eta_mean must be a T ndarray")
        if np.size(eta_var) != self.T:
            raise ValueError("eta_var must be a T ndarray")
        if np.any(eta_var < 0):
            raise ValueError("eta_var must be nonnegative")
        if np.size(mu_0) != self.N:
            raise ValueError("mu_0 must be a N ndarray")
        if np.shape(V_0) != (self.N, self.N):
            raise ValueError("V_0 must be a N by N ndarray")
        if np.any(np.linalg.eigvals(V_0) < 0):
            raise ValueError("V_0 must be nonnegative (definite)")
        return
