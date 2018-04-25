#!/usr/bin/env python

'''
data_generation.py
Author: Nicholas Hoernle
Date: 25 April 2018
Generate data from a range of switching systems.
'''
import numpy as np

def generate_data_1D(num_chains=2, length=1000, switch_prob=0.02, ar=1, **kwargs):
    '''
    num_chains: specify the number of regimes that are present in the data
    length: length of the data
    switch_prob: probability for switching
    chain_params (optional): of length == num_chains, each chains params
    '''

    dim = 1

    def fn1(A, sig):
        if type(A) is list:
            def fn(x):
                return A[0]*x[0] + A[1]*x[1] + np.random.normal(0,np.sqrt(sig))
        else:
            def fn(x):
                return A*x + np.random.normal(0,np.sqrt(sig))
        return fn

    X = []
    if "chain_params" in kwargs:
        assert len(kwargs['chain_params']) == num_chains

        for param in kwargs['chain_params']:
            A, sig = param['A'], param['sigma']
            X.append(fn1(A, sig))

    elif "chain_gen_priors" in kwargs:

        for i in range(num_chains):
            A = np.random.beta(1, kwargs['chain_gen_priors'][0])
            sig = np.random.gamma(1,kwargs['chain_gen_priors'][1])
            X.append(lambda x: A*x + np.random.normal(0,np.sqrt(sig)))

    return generate_data(X, dim, num_chains, length, switch_prob, ar=ar)

def generate_data_nD(dim, num_chains=2, length=1000, switch_prob=0.02, **kwargs):
    '''
    num_chains: specify the number of regimes that are present in the data
    length: length of the data
    switch_prob: probability for switching
    chain_params (optional): of length == num_chains, each chains params
    '''
    def fn1(A, sig):
        def fn(x):
            scale = np.random.gamma(10,2)
            return A.dot(x) + np.random.multivariate_normal(np.zeros(len(A)), sig)
        return fn

    X = []
    if "chain_params" in kwargs:
        assert len(kwargs['chain_params']) == num_chains

        for param in kwargs['chain_params']:
            A, sig = param['A'], param['sigma']
            X.append(fn1(A, sig))

    return generate_data(X, dim, num_chains, length, switch_prob)

def generate_data(X, dim, num_chains, length, switch_prob, ar=1):

    ys = [np.zeros(dim)]
    chains = []
    chain = np.random.choice(np.arange(0,num_chains), p=np.ones(num_chains)/num_chains)
    chains.append(chain)

    if ar > 1:
        ys.append(X[chain]([ys[-1], ys[-1]]))
    else:
        ys.append(X[chain](ys[-1]))

    chains.append(chain)

    for t in range(length-2):

        if np.random.uniform(0,1) < 0.02:
            options = list(range(0, num_chains))
            del options[chain]
            chain = np.random.choice(options)

        if ar > 1:
            ys.append(X[chain]([ys[-1], ys[-2]]))
        else:
            ys.append(X[chain](ys[-1]))
        chains.append(chain)

    return {
            "chains": np.array(chains),
            "Y": np.array(ys)
        }
