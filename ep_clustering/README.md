README for `ep_clustering` python module code.

## Files
* `gibbs.py` provides the `GibbsSampler` class.
* `approx_algorithm.py` provides the `ApproxAlgorithm` base class and `NaiveAlgorithm`, `CollapsedAlgorithm`, and `EPAlgorithm` child classes.
* `likelihoods/` provides the `Likelihood` classes for the various models.
* `data/` provides synthetic dataset generation code.
* `exp_family/` provides exponential family code for Bayesian inference.
* `evaluator/` provides a wrapper class for running and evaluating Gibbs samplers. 
* `kalman_filter/` provides Kalman filter code for the time series models.

This is old (over-engineered) code from the beginning of my PhD that has been ported from python2.7 to python 3+.
There are no guarantees that anything beyond the correlated time series clustering and robust mixture model examples work.

## Usage Example

See `experiments/synthetic_timeseries_clustering_example.py` and `experiments/synthetic_mixture_clustering_example.py` for an overview of the API.

