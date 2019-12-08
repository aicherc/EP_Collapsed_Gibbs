#!/usr/bin/env python
"""
Slice Sampler for vMF likelihood
"""

# Import Modules
import numpy as np
import logging

logger = logging.getLogger(name=__name__)
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )

def bounded(func, lower_bound, upper_bound, out_of_bound_value=-np.inf):
    """ Wrap logf with bounds """
    def bounded_func(x):
        if x < lower_bound:
            return out_of_bound_value
        if x > upper_bound:
            return out_of_bound_value
        else:
            return func(x)
    return bounded_func

class SliceSampler(object):
    """ One Dimensional Slice Sampler

    Based on 'Slice Sampling' by Neal (2003)
    https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461

    Args:
        logf (func): log of target density (up to a constant)
        lower_bound (double): lower bound
        upper_bound (double): upper bound
        interval_method (string): "stepout" or "doubling" (default doubling)
        num_steps (int): number of slice sampler steps for each call to `sample`
        typical_slice_size (double): guess of slice size
        max_interval_size (double): maximum interval size

    """
    def __init__(self, logf, lower_bound=-np.inf, upper_bound=np.inf,
            interval_method="doubling", num_steps=1,
            typical_slice_size=1.0, max_inteval_size=10e5,
            ):
        self.logf = bounded(logf, lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_steps = num_steps
        self.typical_slice_size = typical_slice_size
        self.max_inteval_size = max_inteval_size
        self.interval_method = interval_method
        return

    def sample(self, x_init, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        x_cur = x_init + 0.0

        for step in range(num_steps):
            # Draw Height of slice
            logf_cur = self.logf(x_cur)
            logy = logf_cur + np.log(np.random.rand())

            # Get Interval around slice
            x_l, x_r = self._get_interval(x_cur, logy)

            # Draw New X from Interval containing slice
            x_new = self._draw_from_interval(x_cur, x_l, x_r, logy)

            # Repeat
            x_cur = x_new

        return x_new

    def _get_interval(self, x_cur, logy):
        if self.interval_method == "stepout":
            return self._stepout_interval(x_cur, logy)
        elif self.interval_method == "doubling":
            return self._doubling_interval(x_cur, logy)
        else:
            raise ValueError("Unrecognized interval_method")

    def _stepout_interval(self, x_cur, logy):
        # See Fig. 3 in Slice Sampling by Neal (2003)
        x_l = x_cur - self.typical_slice_size * np.random.rand()
        x_r = x_l + self.typical_slice_size

        m = np.ceil(self.max_inteval_size / self.typical_slice_size)
        J = np.floor(m * np.random.rand())
        K = (m-1)-J

        while (J > 0) and logy < self.logf(x_l):
            x_l = x_l - self.typical_slice_size
            J -= 1

        while (K > 0) and logy < self.logf(x_r):
            x_r = x_r + self.typical_slice_size
            K -= 1

        if J <= 0 or K <= 0:
            logging.warning("inteval may not contain slice")

        return x_l, x_r


    def _doubling_interval(self, x_cur, logy):
        # See Fig. 4 in Slice Sampling by Neal (2003)
        p = np.ceil(np.log2(self.max_inteval_size / self.typical_slice_size))

        x_l = x_cur - self.typical_slice_size * np.random.rand()
        x_r = x_l + self.typical_slice_size

        while((p > 0) and ((logy < self.logf(x_l)) or (logy < self.logf(x_r)))):
            # Expanding either side equally is important for correctness
            if np.random.rand() < 0.5:
                x_l = x_l - (x_r - x_l)
            else:
                x_r = x_r + (x_r - x_l)
            p -= 1

        if p <= 0:
            logging.warning("inteval may not contain slice")

        return x_l, x_r

    def _draw_from_interval(self, x_cur, x_l, x_r, logy):
        """ Draw x_new with logf(x_new) > logy from interval [x_l, x_r] """
        x_new = None
        for _ in range(1000):
            x_candidate = np.random.rand() * (x_r - x_l) + x_l
            logf_candidate = self.logf(x_candidate)

            # If candidate is on slice
            if logf_candidate >= logy:
                x_new = x_candidate
                break

            # Otherwise Shrink Interval
            if x_candidate > x_cur:
                x_r = x_candidate
            elif x_candidate < x_cur:
                x_l = x_candidate
            else:
                raise RuntimeError("Slice shrunk too far")

        if x_new is None:
            raise RuntimeError("Could not find sample on slice")

        return x_new


