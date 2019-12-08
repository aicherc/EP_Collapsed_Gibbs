from ._kalman_filter import KalmanFilter

## Try Importing C++ Implementation
try:
    from ep_clustering.kalman_filter.c_kalman_filter import CKalmanFilter
    _cpp_available = True
except ImportError:
    _cpp_available = False
    CKalmanFilter = None

