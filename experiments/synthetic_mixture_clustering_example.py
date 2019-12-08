# Script to get familiar with the code-base API
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
path_to_fig = "output/synth_mixture_examples/figures"
path_to_data = "output/synth_mixture_examples/data"
if not os.path.isdir(path_to_fig):
    os.makedirs(path_to_fig)
if not os.path.isdir(path_to_data):
    os.makedirs(path_to_data)

######################################################
# Generate Synthetic Data (Robust Clustering)
######################################################
np.random.seed(1234)
mean_scale=3
df=5
df_scale=0.001

data_param = dict(
        num_dim=2,                  # Number of dimensions
        num_obs=100,                # Number of observations
        K=3,                        # Number of clusters
        component_type='student_t', # e.g. 'diag_gaussian', 'mix_gaussian', etc
        component_prior={'mean_sd': np.ones(2)* mean_scale,
            'df_alpha': df/df_scale, 'df_beta': 1.0/df_scale},

)
data_gen = ep.data.MixtureDataGenerator(**data_param)
data = data_gen.generate_data()

# 'data' is a dict containing observations, cluster labels, latent variables and parameters
print(data['matrix']) # Data as a matrix
print(data['z']) # True Cluster Assignments
print(data['parameters'].keys()) # Dictionary of other parameters
print(data['parameters']['cluster_parameters']) # List of dicts of cluster parameters


## Plot and Save Data
def plot_2d_data(data, z=None, x0=0, x1=1, ax=None):
    cp = sns.color_palette("husl", n_colors=data.K)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
    if z is None:
        z = data.z

    for k in range(data.K):
        ind = (z == k)
        ax.plot(data.matrix[ind,x0], data.matrix[ind,x1], "o",
                alpha=0.5, color=cp[k], label="z={0}".format(k))
    ax.legend()
    ax.set_title("X{1} vs X{0}".format(x0, x1))
    return ax

# Plot Data
fig, ax = plt.subplots(1,1)
plot_2d_data(data, ax=ax)
fig.savefig(os.path.join(path_to_fig, "true_data.png"))

# Save Data
joblib.dump(data, filename=os.path.join(path_to_data, "data.p"))


######################################################
# Define Model to Fit
######################################################
## Likelihood
likelihood = ep.construct_likelihood(name="StudentT", data=data)
# See likelihood? for more details (or print(likelihood.__doc__))

## Approximation Algorithm
alg_name = 'EP' # 'naive' or 'collapsed' are the other options
approx_alg = ep.construct_approx_algorithm(
        name=alg_name,
        exp_family=ep.exp_family.NormalWishartFamily,
        damping_factor=0.0, #0.0001,
        separate_likeparams=True,
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
# Only sample z[5] from the posterior
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
# print(sampler.eval_loglikelihood(kind='collapsed')) # Not conjugate so does not exist in closed form


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
        ep.evaluator.metric_function_from_state("z", data.z, "nmi"),
        ep.evaluator.metric_function_from_state("z", data.z, "precision"),
        ep.evaluator.metric_function_from_state("z", data.z, "recall"),
    ]

## Samples to track
# Keep track of theta and cluster assignments z
my_sample_functions = {
        'mean0': lambda sampler: sampler.state.theta[0]['mean'],
        'mean1': lambda sampler: sampler.state.theta[1]['mean'],
        'mean2': lambda sampler: sampler.state.theta[2]['mean'],
        'prec0': lambda sampler: sampler.state.theta[0]['precision'],
        'prec1': lambda sampler: sampler.state.theta[1]['precision'],
        'prec2': lambda sampler: sampler.state.theta[2]['precision'],
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
        plt.close('all')
        plot_metrics(metric_df).savefig(os.path.join(path_to_fig, 'metrics.png'))

        # Save Metrics + Samples
        metric_df.to_pickle(os.path.join(path_to_data, "metrics.p"))
        sample_df.to_pickle(os.path.join(path_to_data, "samples.p"))

## Reload Metrics
#metric_df = pd.read_pickle(os.path.join(path_to_data, "metrics.p"))
#sample_df = pd.read_pickle(os.path.join(path_to_data, "samples.p"))


