# Script to get familiar with the code-base API
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
path_to_fig = "output/synth_mixture_compare_examples/figures"
path_to_data = "output/synth_mixture_compare_examples/data"
if not os.path.isdir(path_to_fig):
    os.makedirs(path_to_fig)
if not os.path.isdir(path_to_data):
    os.makedirs(path_to_data)

######################################################
# Generate Synthetic Data (Robust Clustering)
######################################################
np.random.seed(1234)
mean_scale=5
df=5
df_scale=0.001

data_param = dict(
        num_dim=2,                  # Number of dimensions
        num_obs=600,                # Number of observations
        K=8,                        # Number of clusters
        component_type='student_t', # e.g. 'diag_gaussian', 'mix_gaussian', etc
        component_prior={'mean_sd': np.ones(2)* mean_scale,
            'df_alpha': df/df_scale, 'df_beta': 1.0/df_scale},

)
data_gen = ep.data.MixtureDataGenerator(**data_param)
data = data_gen.generate_data()

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
# Define Samplers to Fit
######################################################
# Student-T-Mixture
likelihood = ep.construct_likelihood(
        name="StudentT",
        data=data,
        df=df,
        )

# Number of Clusters to infer
K=data['K']

# Shared Random Init
init_z = ep.gibbs.random_init_z(N=data['num_obs'], K=K)

# Naive Gibbs
print("Setup Naive Gibbs Sampler")
naive_alg = ep.construct_approx_algorithm(
        name="naive",
        separate_likeparams=True,
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
print("Setup Blocked Gibbs Sampler")
block_alg = ep.construct_approx_algorithm(
        name="collapsed",
        separate_likeparams=True,
        )
block_sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood.deepcopy(),
        approx_alg=block_alg,
        K=K,
        z_prior_type="fixed",
        init_z=init_z,
        )

# EP Gibbs
print("Setup EP Gibbs Sampler")
EP_alg = ep.construct_approx_algorithm(
        name="EP",
        exp_family=ep.exp_family.NormalWishartFamily,
        damping_factor=0.0, #0.0001,
        separate_likeparams=True,
        )
EP_sampler = ep.GibbsSampler(
        data=data,
        likelihood=likelihood.deepcopy(),
        approx_alg=EP_alg,
        K=K,
        full_mcmc=False, # Do not need to sample latent variables u with EP
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
        sampler=block_sampler,
        metric_functions=metric_functions,
        sampler_name="BlockedGibbs",
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
MAX_TIME = 300 # Max time for each algorithm
EVAL_TIME = 30 # Time between evaluations
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


