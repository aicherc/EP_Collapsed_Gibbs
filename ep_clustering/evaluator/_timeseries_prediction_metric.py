"""

Time Series Prediction Metric Function

"""
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(name=__name__)


class TimeSeriesPredictionEvaluator(object):
    """ Class for keeping track of GibbsSampler posterior predictions

    Args:
        new_data (TimeSeriesData): data to predict
        num_samples (int): number of samples to use each evaluation call
        variable_name (string): string of variable name for output
        transformer (func): function to postprocess predictions
    """
    def __init__(self, new_data, num_samples = 5, variable_name = "predict",
            transformer = None):
        self.new_data = new_data
        self.num_samples = num_samples
        self.variable_name = variable_name
        if transformer is not None:
            if not callable(transformer):
                raise ValueError("transfomer must be a callable function")
        self.transformer = transformer

        # Initialize Mean Predict + Number of Observations
        columns = ['mean_predict', 'mean_predict_sq']
        self.predict_df = pd.DataFrame(0.0, columns=columns,
                index=new_data.df.index)
        self.num_obs = 0.0

        self.variable_name = variable_name
        self.metrics = ['rmse', 'mape', 'median ape', '90th ape']
        return

    def sample_predictions(self, sampler):
        """ Use GibbsSampler to generate predictions
        Returns:
            sample_predictions (ndarray): num_rows by num_samples
        """
        sample_predictions = \
            pd.DataFrame(index = self.predict_df.index,
                columns = pd.Index(np.arange(self.num_samples), name="sample"))

        for cluster in range(0, sampler.K):
            # Get cluster membership
            cluster_indices = np.where(sampler.state.z == cluster)[0]
            if cluster_indices.size == 0: # No indices in cluster
                continue

            # Subset series in new_data
            cluster_new_data = self.new_data.subset(cluster_indices)
            row_indices = sample_predictions.index.get_level_values(
                    'observation').isin(cluster_indices)
            if sum(row_indices) == 0: # No rows in new_data
                continue

            # Otherwise Use likelihood to predict
            sampled_y = sampler.likelihood.predict(
                    new_data = cluster_new_data,
                    prior_parameter=sampler.likelihood.theta_prior,
                    num_samples=self.num_samples,
                    )

            sample_predictions.iloc[row_indices] = sampled_y.T
        return sample_predictions

    def _update_mean_prediction(self, samples):
        """ Use GibbsSampler to update mean_predict """
        # Note: mean_predict is of the resid = y - yhat
        n_old = self.num_obs
        n_new = n_old + self.num_samples

        mean_predict = self.predict_df['mean_predict'] * n_old / n_new
        mean_predict += np.sum(samples, axis=1) / n_new

        mean_predict_sq = self.predict_df['mean_predict_sq'] * n_old / n_new
        mean_predict_sq += np.sum(samples**2, axis=1) / n_new

        self.num_obs = n_new
        self.predict_df['mean_predict'] = mean_predict
        self.predict_df['mean_predict_sq'] = mean_predict_sq
        return

    def evaluate(self, sampler):
        """ Metric Function to pass to GibbsSamplerEvaluater
        Returns:
            list of dicts - metrics on the time-series prediction
        """
        logger.info("Sampling Predictions")
        samples = self.sample_predictions(sampler)
        logger.info("Updating Predictions")
        self._update_mean_prediction(samples)

        truth = self.new_data.df[self.new_data.observation_name]
        prediction = self.predict_df['mean_predict']
        if self.transformer is not None:
            truth = self.transformer(truth)
            prediction = self.transformer(prediction)

        resid = prediction - truth
        percent_resid = (prediction - truth) / truth * 100

        metrics = []
        if 'rmse' in self.metrics:
            metrics.append({
                'variable': self.variable_name,
                'metric': 'rmse',
                'value': np.sqrt(np.mean(resid ** 2)),
                })

        if 'mape' in self.metrics:
            metrics.append({
                'variable': self.variable_name,
                'metric': 'mape',
                'value': np.mean(np.abs(percent_resid)),
                })

        if 'median ape' in self.metrics:
            metrics.append({
                'variable': self.variable_name,
                'metric': 'median ape',
                'value': np.median(np.abs(percent_resid)),
                })

        if '90th ape' in self.metrics:
            metrics.append({
                'variable': self.variable_name,
                'metric': '90th ape',
                'value': np.percentile(np.abs(percent_resid), 90),
                })

        return metrics


if __name__ == "__main__":
    print("TimeSeries Prediction Evaluation")

#EOF
