#!/usr/bin/env python

'''
hdp_scripts.py
Author: Nicholas Hoernle
Date: 20 April 2018
A collection of functions that define a Gibbs sampler for a sticky hierarchical Dirichlet process hidden Markov model (see Emily Fox 2008) and the adjusted models for autoregressive and state space model data (see Fox 2009).
'''


import scipy as sp
import pandas as pd
import numpy as np
import time
import pystan
import pickle

Lprime, warmup, L, N = 250, 250, 2**7-1, 10000

exp1_code = '''
data {
    int<lower=1> N;
    vector[N] X;
    vector[N] y;
}

parameters {
    real alpha;
    real beta;
}

model {
    alpha ~ normal(0,10); // change SD to 1 for the tight prior
    beta ~ normal(0,10);  // change SD to 1 for the tight prior
    y ~ normal(alpha + X*beta, 1.2);
}
'''

exp1 = pystan.StanModel(model_code=exp1_code)

def generate_data1(prior_sd=10):
    '''
    Generate data from the linear regression problem
    '''
    N = 15
    X = np.array(np.random.uniform(-10,10, size=N))

    alpha = np.random.normal(0, prior_sd)
    beta = np.random.normal(0, prior_sd)

    Y = np.zeros(N)
    for i in range(N):
        Y[i] = np.random.normal(alpha + X[i]*beta, 1.2)
    return {'alpha': alpha, 'beta': beta, 'X': X, 'y': Y, 'N': N}


def get_advi_param(fit, var):
    return np.array([param for param, name in zip(fit.get('sampler_params'),fit.get('sampler_param_names')) if name == var][0])

def run_experiment_alg1_advi(data_gen, model, variables, alg='meanfield'):
    '''
    Algorithm 1 from the paper
    '''
    result = {}

    # draw a prior sample and simulated dataset
    exp_data = data_gen()

    # draw L posterior samples
    fit = model.vb(data=exp_data, output_samples=L, algorithm=alg)

    # for each variable follow Alg
    for k in variables:
        samples = get_advi_param(fit, k)
        theta_l = samples
        rankf = np.sum(theta_l < exp_data[k])
        result[k] = rankf
    return result

N_ = 10000

rank_alphamf = []
rank_betamf = []

for i in range(N_):
    res = run_experiment_alg1_advi(generate_data1, exp1, ['alpha', 'beta'])
    rank_alphamf.append(res['alpha'])
    rank_betamf.append(res['beta'])

rank_alphafr = []
rank_betafr = []
for i in range(N_):
    res = run_experiment_alg1_advi(generate_data1, exp1, ['alpha', 'beta'], 'fullrank')
    rank_alphafr.append(res['alpha'])
    rank_betafr.append(res['beta'])

results = {
    'alpha_mf': rank_alphamf,
    'beta_mf': rank_betamf,
    'alpha_fr': rank_alphafr,
    'beta_fr': rank_betafr
}

pickle.dump(results, open('advi_results.pkl', 'wb'))
print('done')
