"""

Gibbs Sampler Evaluation Wrapper

"""

import pandas as pd
import time
import logging
logger = logging.getLogger(name=__name__)


class GibbsSamplerEvaluater(object):
    """ Wrapper to handle measuring a GibbsSampler's performance

    Example:

    Args:
        sampler (GibbsSampler): the Gibbs sampler
        metric_functions (func or list of funcs): evaluation functions
            Each function takes a sampler and returns a dict (or list of dict)
                {metric, variable, value} for each
            See ep_clustering.evaluator.metric_function_from_state or
                ep_clustering.evaluator.metric_function_from_sampler
        sample_functions (dict of funcs, optional): samples to save
            `key` is the name of column, `value` is a function with
                input: sampler
                output: obj saved under column `key`
        sampler_name (string, optional): name for sampler
        data_name (string, optional): name for data

    Attributes:
        metrics (pd.DataFrame): output data frame with columns
            * metric (string)
            * variable (string)
            * value (double)
            * iteration (int)

    Methods:
        get_metrics(self, **kwargs)
        reset_metrics(self)
        save_metrics(self, file_name)

        get_samples(self)
        reset_samples(self)
        save_samples(self, file_name)

        evaluate_sampler_step(self, sampler_func_name, sampler_func_kwargs)
        evaluate_metric_functions(self)

    """
    def __init__(self, sampler, metric_functions,
            sample_functions = {},
            sampler_name = None, data_name = None,
            metric_log_level = logging.INFO):
        self.sampler = sampler
        self.metric_log_level = metric_log_level

        # Check metric Functions
        self._process_metric_functions(metric_functions)
        self.metric_functions = metric_functions

        if not isinstance(sample_functions, dict):
            raise ValueError("sample_functions must be a dict of str:func")
        self.sample_functions = sample_functions

        if sampler_name is None:
            sampler_name = sampler.name
        self.sampler_name = sampler_name

        if data_name is None:
            data_name = sampler.likelihood.data.get("name", "")
        self.data_name = data_name

        self.iteration = 0
        self._start_time = time.time()
        self._init_metrics()
        self._init_samples()
        return

    @staticmethod
    def _process_metric_functions(metric_functions):
        if callable(metric_functions):
            metric_functions = [metric_functions]

        elif isinstance(metric_functions, list):
            for metric_function in metric_functions:
                if not callable(metric_function):
                    raise ValueError("metric_functions must be list of funcs")
        else:
            ValueError("metric_functions should be list of funcs")

    def _init_metrics(self):
        self.metrics = pd.DataFrame()
        self.eval_metric_functions()
        init_metric = {
            "variable": "time",
            "metric": "time",
            "value": 0.0,
            "iteration": self.iteration,
            }
        self.metrics = self.metrics.append(init_metric, ignore_index = True)
        return

    def get_metrics(self, extra_columns={}):
        """ Return a pd.DataFrame copy of metrics

        Args:
            extra_columns (dict): extra metadata to add as columns
        Returns:
            pd.DataFrame with columns
                metric, variable, value, iteration, sampler, data, extra_columns

        """
        metrics = self.metrics.copy()
        metrics["sampler"] = self.sampler_name
        metrics["data"] = self.data_name
        for k,v in extra_columns.items():
            metrics[k] = v
        return metrics

    def reset_metrics(self):
        """ Reset self.metrics """
        logger.info("Resetting metrics")
        self.iteration = 0
        self._init_metrics()
        return

    def save_metrics(self, filename, extra_columns = {}):
        """ Save a pd.DataFrame to filename + '.csv' """
        metrics = self.get_metrics(extra_columns)

        logger.info("Saving metrics to file %s", filename)
        metrics.to_csv(filename + ".csv", index = False)
        return

    def evaluate_sampler_step(self, sampler_func_name = "one_step",
            sampler_func_kwargs = None):
        """ Evaluate the performance of the sampler steps

        Args:
            sampler_func_name (string or list of strings):
                name(s) of sampler member functions
                (e.g. `one_step` or `['sample_z', 'sample_z']`)
            sampler_func_kwargs (kwargs or list of kwargs):
                options to pass to sampler_func_name

        """
        logger.info("Sampler %s, Iteration %d",
                self.sampler_name, self.iteration+1)

        # Single Function
        if isinstance(sampler_func_name, str):
            sampler_func = getattr(self.sampler, sampler_func_name, None)
            if sampler_func is None:
                raise ValueError(
                    "sampler_func_name `{}` is not in sampler".format(
                            sampler_func_name)
                        )
            if sampler_func_kwargs is None:
                sampler_func_kwargs = {}

            sampler_start_time = time.time()
            sampler_func(**sampler_func_kwargs)
            sampler_step_time = time.time() - sampler_start_time

        # Multiple Steps
        elif isinstance(sampler_func_name, list):
            sampler_funcs = [getattr(self.sampler, func_name, None)
                    for func_name in sampler_func_name]
            if None in sampler_funcs:
                raise ValueError("Invalid sampler_func_name")

            if sampler_func_kwargs is None:
                sampler_func_kwargs = [{} for _ in sampler_funcs]
            if not isinstance(sampler_func_kwargs, list):
                raise TypeError("sampler_func_kwargs must be a list of dicts")
            if len(sampler_func_kwargs) != len(sampler_func_name):
                raise ValueError("sampler_func_kwargs must be same length " +
                    "as sampler_func_name")
            sampler_start_time = time.time()
            for sampler_func, kwargs in zip(sampler_funcs, sampler_func_kwargs):
                sampler_func(**kwargs)
            sampler_step_time = time.time() - sampler_start_time

        else:
            raise TypeError("Invalid sampler_func_name")

        self.iteration += 1
        time_metric = {
            "variable": "time",
            "metric": "time",
            "value": sampler_step_time,
            "iteration": self.iteration,
            }
        self.metrics = self.metrics.append(time_metric, ignore_index = True)
        self.eval_metric_functions()

        # Save Samples
        if self.sample_functions:
            self._eval_samples()

        return

    def eval_metric_functions(self, metric_functions = None):
        """ Evaluate the state of the sampler

        Args:
           metric_functions (list of funcs): evaluation functions
            Defaults to metric functions defined in __init__

        """
        if metric_functions is None:
            metric_functions = self.metric_functions
        self._process_metric_functions(metric_functions)

        iter_metrics = []
        for metric_function in metric_functions:
            metric = metric_function(self.sampler)
            log_level = getattr(metric_function, "log_level",
                    self.metric_log_level)
            if isinstance(metric, dict):
                logger.log(log_level, "Metric: %s", str(metric))
                iter_metrics.append(metric)
            elif isinstance(metric, list):
                for met in metric:
                    if not isinstance(met, dict):
                        raise TypeError("Metric must be dict or list of dict")
                    logger.log(log_level, "Metric: %s", str(met))
                    iter_metrics.append(met)
            else:
                raise TypeError("Metric must be dict or list of dict")

        iter_metrics = pd.DataFrame(iter_metrics)
        iter_metrics["iteration"] = self.iteration

        self.metrics = self.metrics.append(iter_metrics, ignore_index = True)
        return

    def _init_samples(self):
        columns = ["iteration"]
        columns.extend(self.sample_functions.keys())
        self.samples = pd.DataFrame(columns = columns)
        self._eval_samples()
        return

    def get_samples(self):
        """ Return a pd.DataFrame of samples """
        if not self.sample_functions:
            logger.warning("No sample functions were provided to track!!!")
        samples = self.samples.copy()
        samples["sampler"] = self.sampler_name
        samples["data_name"] = self.data_name
        return samples

    def reset_samples(self):
        """ Reset self.metrics """
        logger.info("Resetting samples")
        self._init_samples()
        return

    def save_samples(self, filename):
        """ Save a pd.DataFrame to filename + '.csv' """
        samples = self.get_samples()

        logger.info("Saving samples to file %s", filename)
        samples.to_csv(filename + ".csv", index = False)
        return

    def _eval_samples(self):
        # Return a dict of current tracked samples
        samples = { key : value(self.sampler) + 0.0
                for key, value in self.sample_functions.items() }
        samples.update({"iteration": self.iteration})
        self.samples = self.samples.append(samples, ignore_index=True)
        return

