# Refresher script to get familiar with the code-base API
import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import ep_clustering as ep
from tqdm import tqdm # Progress Bar

# Make Paths to output/figures
path_to_fig = "output/synth_ts_examples/figures"
path_to_data = "output/synth_ts_examples/data"
if not os.path.isdir(path_to_fig):
    os.makedirs(path_to_fig)
if not os.path.isdir(path_to_data):
    os.makedirs(path_to_data)



######################################################
# Generate Synthetic Data
######################################################
np.random.seed(12345)
data_param = dict(
        num_dim=100,        # Length of each time series
        num_obs=50,         # Number of time series
        K=3,                # Number of clusters
        sigma2_x = 0.01,    # Latent noise variance
        #missing_obs = 0.05, # Fraction of observations missing (i.e. NaNs)
)
data_gen = ep.data.TimeSeriesDataGenerator(**data_param)
data = data_gen.generate_data()

# 'data' is a dict containing observations, cluster labels, latent variables and parameters
print(data['df']) # Data as a pd.Series in tall-format
# To convert data['df'] to a matrix, use 'ep._utils.convert_df_to_matrix'
Y_mat, missing_mat = ep._utils.convert_df_to_matrix(data['df'])
print(Y_mat)
# Convert back with 'ep._utils.convert_matrix_to_df'
df = ep._utils.convert_matrix_to_df(Y_mat)
print(df)


print(data['z']) # True Cluster Assignments
print(data['theta']) # Cluster Means / Latent Factor Means
print(data['parameters'].keys()) # Dictionary of other parameters
print(data['parameters']['x']) # Latent time series


## Plot and Save Data
def plot_truth_data(data):
    fig, ax = plt.subplots(1,1)
    plot_df = data.df.reset_index()
    z_df = pd.DataFrame({'truth': data.z,
        'observation': np.arange(len(data.z))})
    plot_df = pd.merge(plot_df, z_df)
    sns.lineplot(x='dimension', y='y', hue='truth',
            palette=sns.color_palette('deep', data['K']),
            alpha=0.5,
            units='observation', estimator=None, data=plot_df,
            ax=ax)
    return fig, ax

# Plot Data

fig, ax = plot_truth_data(data)
fig.savefig(os.path.join(path_to_fig, "true_data.png"))
# Save Data
joblib.dump(data, filename=os.path.join(path_to_data, "data.p"))


######################################################
# Define Model to Fit
######################################################
## Likelihood
likelihood = ep.construct_likelihood(name="TimeSeries", data=data)
# See likelihood? for more details (or print(likelihood.__doc__))

## Approximation Algorithm
alg_name = 'EP' # 'naive' or 'collapsed' are the other options
approx_alg = ep.construct_approx_algorithm(
        name=alg_name,
        exp_family=ep.exp_family.DiagNormalFamily,
    )
# See approx_alg? for more details (or print(approx_alg.__doc__))

## Number of Clusters to Infer
Kinfer = 3

## Setup Approx Gibbs Sampler
sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood,
        K=Kinfer,
        approx_alg=approx_alg,
        )
# See ep.GibbsSampler? for more details on how to set prior and other algorithm options
# See sampler? for more details (or print(sampler.__doc__))

print(sampler.state) # Initial state of latent vars 'z' and parameters
print(sampler.options) # Gibbs sampling options (including prior)

# Randomly initialize the inital_state
sampler.init_state()
# sampler.init_state(z=data['z']) # init from ground truth labels

# Sample z from the posterior (based on alg_name)
print(sampler.sample_z())
# Only sample z[5] from the posterior, returns [5], old_z[5], new_z[5]
print(sampler.sample_one_z(5))
# Note, the EP approx parameters are updated automatically whenever z is sampled

# Sample Theta
print(sampler.sample_theta())

# Sample Other Likelihood Parameters
sampler.sample_likelihood_parameters()

# Helper Function to Sample Z, Theta, and Likelihood Parameters using Blocked Gibbs
sampler.one_step(full_mcmc=True)
# If full_mcmc is False, then the likelihood_parameters will not be sampled

# Manually update all EP Approximation Parameters (for Full, exact EP)
sampler.update_approx_alg()

# Evaluate log Pr(y | z, theta, likelihood parameters) of data
print(sampler.eval_loglikelihood(kind='naive'))

# Evaluate log Pr(y | z, likelihood parameters) of data
print(sampler.eval_loglikelihood(kind='collapsed'))


######################################################
# Define Evaluation Metrics
######################################################
## Metrics to track
# -Loglikelihood
# -NVI (normalized variation of information) a divergence metric between clusters
# -MSE of parameters likelihood parameters and latent series x
my_metric_functions = [
        ep.evaluator.metric_function_from_sampler("eval_loglikelihood"),
        ep.evaluator.metric_function_from_state("z", data.z, "nvi"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "A"],
            data.parameters.A, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "x"],
            data.parameters.x, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "lambduh"],
            data.parameters.lambduh, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "sigma2_x"],
            data.parameters.sigma2_x, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "sigma2_y"],
            data.parameters.sigma2_y, "mse"),
    ]

## Samples to track
# Keep track of theta and cluster assignments z
my_sample_functions = {
        'theta': lambda sampler: sampler.state.theta,
        'z': lambda sampler: sampler.state.z,
    }

## Construct Sampler Wrapper
my_evaluator = ep.GibbsSamplerEvaluater(
        sampler=sampler,
        metric_functions=my_metric_functions,
        sample_functions=my_sample_functions,
        sampler_name="example",
    )


######################################################
# Run Sampler with Online Evaluation
######################################################
from tqdm import tqdm # Progress Bar
for _ in tqdm(range(10)):
    my_evaluator.evaluate_sampler_step()
metric_df = my_evaluator.get_metrics()
sample_df = my_evaluator.get_samples()

print(metric_df) # Metrics for first iterations are pd.DataFrame
print(sample_df) # Samples for first iterations are pd.DataFrame

# For example to look at just the loglikelihood
print(metric_df.query('metric == "eval_loglikelihood"'))

# Metric Plotter
def plot_metrics(metric_df):
    metric_df['variable'] = metric_df['variable'].apply(
            lambda x: x.strip('likelihood_parameter'))
    metric_df['metric_var'] = metric_df['metric'] + "_" + metric_df['variable']
    g = sns.FacetGrid(
            data=metric_df, col="metric_var", col_wrap=4, sharey=False,
            )
    g = g.map(plt.plot, "iteration", "value")
    return g


# Run for 100 EPOCHs
num_epochs = 100 # number of iterations (passes over the data set)
for _ in tqdm(range(num_epochs)):
    my_evaluator.evaluate_sampler_step()
    # Checkpoint Metrics + Samples every 10 Epochs
    if _ % 10 == 0:
        metric_df = my_evaluator.get_metrics()
        sample_df = my_evaluator.get_samples()
        # Plot Metrics over time
        plot_metrics(metric_df).savefig(os.path.join(path_to_fig, 'metrics.png'))

        # Save Metrics + Samples
        metric_df.to_pickle(os.path.join(path_to_data, "metrics.p"))
        sample_df.to_pickle(os.path.join(path_to_data, "samples.p"))


######################################################
# Full Example of TS with regression covariates
######################################################
data_param = dict(
        num_dim=150,        # Length of each time series
        num_obs=100,        # Number of time series
        K=4,               # Number of clusters
        sigma2_x = 0.01,    # Latent noise variance
        #missing_obs = 0.05, # Fraction of observations missing (i.e. NaNs)
        A=np.ones(100) * 0.95,
        regression=True, # Include Covariates
        covariate_coeff=10*np.ones((100, 3)), # Covariates

)
data_gen = ep.data.TimeSeriesDataGenerator(**data_param)
data = data_gen.generate_data()
print(data['df'].head())

# Save Data
joblib.dump(data, filename=os.path.join(path_to_data, "tsreg_data.p"))


Kinfer = 4
covariate_names = [col for col in data['df'].columns if "cov" in col]

likelihood = ep.construct_likelihood(name="TimeSeriesRegression",
        data=data, covariate_names=covariate_names)
approx_alg = ep.construct_approx_algorithm(
        name='EP',
        exp_family=ep.exp_family.DiagNormalFamily,
    )
sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood,
        K=Kinfer,
        approx_alg=approx_alg,
        )
sampler.init_state()

my_metric_functions = [
        ep.evaluator.metric_function_from_sampler("eval_loglikelihood"),
    ]
my_metric_functions += [
        ep.evaluator.metric_function_from_state("z", data.z, "nvi"),
        ep.evaluator.metric_function_from_state("z", data.z, "nmi"),
        ep.evaluator.metric_function_from_state("z", data.z, "precision"),
        ep.evaluator.metric_function_from_state("z", data.z, "recall"),
    ]
my_metric_functions += [
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "A"],
            data.parameters.A, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "x"],
            data.parameters.x, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "lambduh"],
            data.parameters.lambduh, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "sigma2_x"],
            data.parameters.sigma2_x, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "sigma2_y"],
            data.parameters.sigma2_y, "mse"),
        ep.evaluator.metric_function_from_state(
            ['likelihood_parameter', "covariate_coeff"],
            data.parameters.covariate_coeff, "mse"),
        ]

my_sample_functions = {
        'theta': lambda sampler: sampler.state.theta,
        'z': lambda sampler: sampler.state.z,
    }

# Construct Evaluator
evaluator = ep.GibbsSamplerEvaluater(
        sampler=sampler,
        metric_functions=my_metric_functions,
        sample_functions=my_sample_functions,
        sampler_name="regression_example",
        )

# Run for 100 EPOCHs
num_epochs = 100 # number of iterations (passes over the data set)
for _ in tqdm(range(num_epochs)):
    evaluator.evaluate_sampler_step(['one_step'])
    # Checkpoint Metrics + Samples every 10 Epochs
    if _ % 10 == 0:
        metric_df = evaluator.get_metrics()
        sample_df = evaluator.get_samples()
        # Plot Metrics over time
        plt.close('all')
        plot_metrics(metric_df).savefig(os.path.join(path_to_fig, 'tsreg_metrics.png'))

        # Save Metrics + Samples
        metric_df.to_pickle(os.path.join(path_to_data, "tsreg_metrics.p"))
        sample_df.to_pickle(os.path.join(path_to_data, "tsreg_samples.p"))



