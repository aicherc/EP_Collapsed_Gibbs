#!/usr/bin/env python
"""

Gibbs Sampler Data Class

"""

import numpy as np
import pandas as pd
import logging
from ep_clustering._utils import Map

# Author Information
__author__ = "Christopher Aicher"

# Modify the root logger
logger = logging.getLogger(name=__name__)


class GibbsData(Map):
    """ Data for GibbsSampler

    Must contain `num_obs` and `num_dim` attributes

    See ep_clustering.data.generate_data to generate data

    Methods:
        _validate_data(self): check that object has proper attributes

    """
    def __init__(self, *args, **kwargs):
        super(GibbsData, self).__init__(*args, **kwargs)
        self._validate_data()
        return

    def _validate_data(self):
        # Check Gibbs Data has required attributes
        if "num_obs" not in self:
            raise ValueError("`num_obs` must be defined in GibbsData")
        if "num_dim" not in self:
            raise ValueError("`num_dim` must be defined in GibbsData")
        return

def _categorical_sample(probs):
    """ Draw a categorical random variable over {0,...,K-1}
    Args:
      probs (K ndarray) - probability of each value
    Returns:
      draw (int) - random outcome
    """
    return int(np.sum(np.random.rand(1) > np.cumsum(probs)))


# EOF
