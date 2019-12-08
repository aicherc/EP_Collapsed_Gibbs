# Compare the EP Gibbs sampler to classic Gibbs Samplers
# Print (Warning this is a long)
import matplotlib as mpl
mpl.use('Agg')

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import ep_clustering as ep
from tqdm import tqdm # Progress Bar

# Make Paths to output/figures
path_to_fig = "output/synth_ts_compare_examples/figures"
path_to_data = "output/synth_ts_compare_examples/data"
if not os.path.isdir(path_to_fig):
    os.makedirs(path_to_fig)
if not os.path.isdir(path_to_data):
    os.makedirs(path_to_data)

######################################################
# Generate Synthetic Data (Robust Clustering)
######################################################
np.random.seed(12345)
data_param = dict(
        num_dim=200,        # Length of each time series
        num_obs=300,         # Number of time series
        K=20,                # Number of clusters
        sigma2_x = 0.01,    # Latent noise variance
        #missing_obs = 0.05, # Fraction of observations missing (i.e. NaNs)
)
data_param['A'] = 0.95 * np.ones(data_param['num_obs'])
data_gen = ep.data.TimeSeriesDataGenerator(**data_param)
data = data_gen.generate_data()

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
# Define Samplers to Fit
######################################################
# TimeSeries Likelihood
likelihood = ep.construct_likelihood(
        name="TimeSeries",
        data=data,
        )

# Number of Clusters to infer
K=data['K']

# Shared Random Init
init_z = ep.gibbs.random_init_z(N=data['num_obs'], K=K)

# Naive Gibbs
print("Setup Naive Gibbs Sampler")
naive_alg = ep.construct_approx_algorithm(
        name="naive",
        )
naive_sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood.deepcopy(),
        approx_alg=naive_alg,
        K=K,
        z_prior_type="fixed",
        init_z=init_z,
        )

# Blocked Gibbs
print("Setup Collapsed Gibbs Sampler")
collapsed_alg = ep.construct_approx_algorithm(
        name="collapsed",
        )
collapsed_sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood.deepcopy(),
        approx_alg=collapsed_alg,
        K=K,
        z_prior_type="fixed",
        init_z=init_z,
        )

# EP Gibbs
print("Setup EP Gibbs Sampler")
EP_alg = ep.construct_approx_algorithm(
        name="EP",
        exp_family=ep.exp_family.DiagNormalFamily,
        )
EP_sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood.deepcopy(),
        approx_alg=EP_alg,
        K=K,
        z_prior_type="fixed",
        init_z=init_z,
        )

######################################################
# Define Evaluation
######################################################
metric_functions = [
            ep.evaluator.metric_function_from_sampler("eval_loglikelihood"),
            ep.evaluator.metric_function_from_state("z", data.z, "nvi"),
            ep.evaluator.metric_function_from_state("z", data.z, "nmi"),
            ep.evaluator.metric_function_from_state("z", data.z, "precision"),
            ep.evaluator.metric_function_from_state("z", data.z, "recall"),
            ]

evaluators = [
    ep.GibbsSamplerEvaluater(
        sampler=naive_sampler,
        metric_functions=metric_functions,
        sampler_name="NaiveGibbs",
        data_name="simple_example",
        ),
    ep.GibbsSamplerEvaluater(
        sampler=collapsed_sampler,
        metric_functions=metric_functions,
        sampler_name="CollapsedGibbs",
        data_name="simple_example",
        ),
    ep.GibbsSamplerEvaluater(
        sampler=EP_sampler,
        metric_functions=metric_functions,
        sampler_name="EPGibbs",
        data_name="simple_example",
        ),
    ]

######################################################
# Comparison Plot Helper Functions
######################################################

def plot_metric_vs_iteration(evaluators, metric="nmi"):
    fig, ax = plt.subplots(1,1)
    max_iter = []
    for evaluator in evaluators:
        nmi = evaluator.metrics.query('metric == "nmi"')
        ax.plot(nmi.iteration, nmi.value, label=evaluator.sampler_name)
        max_iter.append(nmi.iteration.max())
    ax.set_xlim([0, min(max_iter)])
    ax.set_xlabel('Iteration (epoch)')
    ax.set_ylabel(metric)
    ax.legend()
    return fig, ax


def plot_metric_vs_time(evaluators, metric="nmi"):
    fig, ax = plt.subplots(1,1)
    max_time = []
    for evaluator in evaluators:
        nmi = evaluator.metrics.query('metric == "nmi"')
        time = evaluator.metrics.query('metric == "time"')
        ax.plot(time.value.cumsum(), nmi.value, label=evaluator.sampler_name)
        max_time.append(time.value.sum())
    ax.set_xlim([0, min(max_time)])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel(metric)
    ax.legend()
    return fig, ax


######################################################
# Run Each Method
######################################################
MAX_TIME = 3600 # Max time for each algorithm (1 hour)
EVAL_TIME = 120 # Time between evaluations (2 min)
evaluator_times = [0 for _ in evaluators]

print("Running Each Sampler")
while np.any([evaluator_time < MAX_TIME for evaluator_time in evaluator_times]):
    for ii in range(len(evaluators)):
        if evaluator_times[ii] >= MAX_TIME:
            continue
        start_time = time.time()
        while (time.time() - start_time) < EVAL_TIME:
            evaluators[ii].evaluate_sampler_step(['one_step'])
        evaluator_times[ii] += time.time()-start_time
        inference_time = evaluators[ii].metrics.query("metric == 'time'")['value'].sum()
        print("Sampler {0}, inference time {1}, total time {2}".format(
            evaluators[ii].sampler_name, inference_time, evaluator_times[ii])
            )
        evaluators[ii].get_metrics().to_pickle(os.path.join(path_to_data,
            "{0}_metrics.p".format(evaluators[ii].sampler_name)))

    # Comparision Plots
    plt.close('all')
    plot_metric_vs_time(evaluators, metric='nmi')[0].savefig(
            os.path.join(path_to_fig, 'nmi_vs_time.png'))
    plot_metric_vs_iteration(evaluators, metric='nmi')[0].savefig(
            os.path.join(path_to_fig, 'nmi_vs_iter.png'))


