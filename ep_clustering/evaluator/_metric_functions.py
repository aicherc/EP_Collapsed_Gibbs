"""

Gibbs Sampler Metric Functions

"""
import numpy as np
import ep_clustering._utils as _utils
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score

def metric_function_from_sampler(sampler_func_name, metric_name=None,
        **sampler_func_kwargs):
    """ Returns metric function that evaluates sampler_func_name

    Example:
        metric_function_of_sampler(sampler_func_name = "loglikelihood",
                          kind = "naive")
    """
    if metric_name is None:
        metric_name = sampler_func_name

    def custom_metric_function(sampler):
        sampler_func = getattr(sampler, sampler_func_name, None)
        if sampler_func is None:
            raise ValueError(
                "sampler_func_name `{}` is not in sampler".format(
                        sampler_func_name)
                    )
        else:
            metric_value = sampler_func(**sampler_func_kwargs)
            metric = {'variable': 'sampler',
                      'metric': metric_name,
                      'value': metric_value}
        return metric
    return custom_metric_function

def metric_function_from_state(state_variable_name, target_value, metric_name,
        return_variable_name=None, state_variable_func=None, log_level=None):
    """ Returns metric function that compares samplers' state to a target

    If the variable is a list of variables, then

    Args:
        state_variable_name (string, list): name of variable in sampler.state
            (e.g. 'z' or ['likelihood_parameter', 'A'])
        target_value (ndarray or list of ndarrays):
            target value(s) for sampler.state.variable_name
        metric_name (string): name of a metric function
            * 'nvi': normalized variation of information
            * 'nmi': normalized mutual information
            * 'precision': cluster precision
            * 'recall': cluster recall
            * 'mse': mean squared error
            * 'mae': mean absolute error
            * 'l2_percent_error': percent_error
            (see construct_metric_function)
        return_variable_name (string, optional): name of metric return name
            default is `state_variable_name` (concat by "_" if list)
        state_variable_func (func, optional):
            function to get variable of interest from state
            (e.g. state_variable_func = lambda state: state.z)
            Default is inferred from `state_variable_name`
        log_level (int, optional): logging level (e.g. logging.INFO)

    Returns:
        A function that returns dictionary of
            return_variable_name, metric, value
        If comparing list of values,
        then the returnvariable name has suffix `_[#]` for its position

    """
    metric_func = construct_metric_function(metric_name)
    if return_variable_name is None:
        if isinstance(state_variable_name, list):
            return_variable_name = "_".join(state_variable_name)
        else:
            return_variable_name = state_variable_name
    if state_variable_func is not None:
        if not callable(state_variable_func):
            raise ValueError("`state_variable_func` must be callable")

    def custom_metric_function(sampler):
        if state_variable_func is not None:
            state_variable = state_variable_func(sampler.state)
        else:
            if isinstance(state_variable_name, list):
                state_variable = _utils.getFromNestDict(sampler.state,
                        state_variable_name)
            else:
                state_variable = sampler.state[state_variable_name]

        if isinstance(state_variable, list):
            if not isinstance(target_value, list):
                raise TypeError(
                        "target_value must be a list to match {0}".format(
                            str(state_variable_name))
                        )
            metric = []
            for ii, (state_var, target_var) in enumerate(zip(
                state_variable, target_value)):
                metric_value = metric_func(state_var, target_var)
                metric.append(
                    {'variable': return_variable_name + "_" + str(ii),
                      'metric': metric_name,
                      'value': metric_value,
                      })
        else:
            metric_value = metric_func(state_variable, target_value)
            metric = {'variable': return_variable_name,
                      'metric': metric_name,
                      'value': metric_value
                      }
        return metric

    # Helper to assign log_level
    if log_level is not None:
        custom_metric_function.log_level = log_level

    return custom_metric_function

def construct_metric_function(metric_name):
    """ Return a metric function

    Args:
        metric_name (string): name of metric. Must be one of
            * 'nvi': normalized variation of information
            * 'nmi': normalized mutual information
            * 'precision': cluster precision
            * 'recall': cluster recall
            * 'mse': mean squared error
            * 'mae': mean absolute error
            * 'l2_percent_error': percent_error

    Returns:
        metric_function (function):
            function of two inputs (result, expected)
    """
    if(metric_name == "mse"):
        def metric_function(result, expected):
            return np.mean((result - expected)**2)
        return metric_function

    elif(metric_name == "nvi"):
        def metric_function(result, expected):
            return var_info(result, expected, normalized=True)
        return metric_function

    elif(metric_name == "nmi"):
        def metric_function(predict, truth):
            return normalized_mutual_info_score(truth, predict, average_method='geometric')
        return metric_function

    elif(metric_name == "precision"):
        def metric_function(predict, truth):
            cm = confusion_matrix(truth, predict)
            return np.sum(np.max(cm, axis=0))/(np.sum(cm)*1.0)
        return metric_function

    elif(metric_name == "recall"):
        def metric_function(predict, truth):
            cm = confusion_matrix(truth, predict)
            return np.sum(np.max(cm, axis=1))/(np.sum(cm)*1.0)
        return metric_function

    elif(metric_name == "mae"):
        def metric_function(result, expected):
            return np.mean(np.abs(result - expected))
        return metric_function

    elif(metric_name == "l2_percent_error"):
        def metric_function(result, expected):
            return np.linalg.norm(result-expected)/np.linalg.norm(expected)*100
        return metric_function
    else:
        raise ValueError("Unrecognized metric name = %s" % metric_name)


def heldout_loglikelihood(test_set, variable_name = 'test_set'):
    """ Returns metric function that evaluates heldout_likelihood of test_set

    Args:
        test_set (array-like): data to evaluate
        variable_name (string): name of `variable` in returned dict

    Returns:
        metric_function (function):
            function of sampler that returns dict

    Example:
        heldout_loglikelihood(test_set)
    """
    def heldout_loglikelihood_metric_func(sampler):
        heldout_loglikelihood = 0.0
        for test_data in test_set:
            for k in range(sampler.K):
                #TODO FIX THIS HACK
                likelihood = sampler.approx_alg.get_likelihood(k)
                original_data = likelihood.y[0] + 0.0
                likelihood.y[0] = test_data
                component_loglikelihood = \
                    sampler.approx_alg.get_likelihood(k).loglikelihood(
                            index = -1,
                            theta = sampler.state.theta[k],
                            )
                likelihood.y[0] = original_data

                heldout_loglikelihood += (
                        component_loglikelihood * np.mean(sampler.state.z == k)
                        )

        heldout_loglikelihood *= 1.0/len(test_set)

        metric = {
                'metric':'heldout_loglikelihood',
                'variable': variable_name,
                'value': heldout_loglikelihood}
        return metric

    return heldout_loglikelihood_metric_func


# HELPER FUNCTIONS
def var_info(z1, z2, normalized=False):
    """ Variation of Information

    Args:
      z1 (N ndarray): cluster assignments
      z2 (N ndarray): cluster assignments
      normalized (bool): normalize by joint entropy (default false)

    Returns:
      vi (double >= 0): divergence between the two assignments
    """
    n = len(z1)
    p1 = np.unique(z1)
    p2 = np.unique(z2)
    r = np.zeros((len(p1), len(p2)))
    for i1, v1 in enumerate(p1):
        for i2, v2 in enumerate(p2):
            r[i1,i2] = np.sum((z1 == v1) & (z2 == v2))*1.0/n
    r1 = np.sum(r, axis=1)
    r2 = np.sum(r, axis=0)
    entropy_1 = -1.0*np.sum(r1[r1 > 0] * np.log(r1[r1 > 0]))
    entropy_2 = -1.0*np.sum(r2[r2 > 0] * np.log(r2[r2 > 0]))
    prod = np.outer(r1, r2)
    mutual_info = np.sum(r[r>0] * (np.log(r[r > 0]) - np.log(prod[r > 0])))
    vi = entropy_1 + entropy_2 - 2.0*mutual_info
    if not normalized:
        return vi
    else:
        joint_entropy = -1.0*np.sum(r[r > 0] * np.log(r[r>0]))
        nvi = vi/joint_entropy
        return nvi
