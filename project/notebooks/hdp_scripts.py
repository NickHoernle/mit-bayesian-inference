#!/usr/bin/env python

'''
hdp_scripts.py
Author: Nicholas Hoernle
Date: 20 April 2018
A collection of functions that define a Gibbs sampler for a sticky hierarchical Dirichlet process hidden Markov model (see Emily Fox 2008) and the adjusted models for autoregressive and state space model data (see Fox 2009).
'''

import scipy.stats as stats
import scipy as sp
import pandas as pd
import numpy as np
import time

# custom scripts:
import classification_evaluation as hdp_eval
from multivariate_normal import MultivariateNormal

#=============================================================
# backward pass of the data
#=============================================================
def backward_step(obs, likeihood_fn, pi, m_tplus1, L, **kwargs):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = np.zeros((L,L), dtype=np.float128)
    messages += (np.log(pi) + likeihood_fn.logpdf(obs) + m_tplus1)
    return sp.misc.logsumexp(messages, axis=1)

def backward_algorithm(Y, mean_func, cov_func, likelihood, params, **kwargs):
    '''
    Calculate the backward messages for all time T...1
    '''

    pi = params['pi']
    theta = params['theta']
    L,T = params['L'],params['T']


    # we have L models and T time steps
    bkwd = np.zeros(shape=(T+1, L), dtype=np.float128)
    # (a) set the messages T+1,T t
    bkwd[-1,:] = 1/L

    # (b) compute the backward messages
    for tf, yt in enumerate(Y):

        t = (T-1)-tf # we need the reverse index

        mu_k_j = [[mean_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
        sigma_k_j = [[cov_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
        # print(mu_k_j)
        likelihood_fn = likelihood(mu_k_j, sigma_k_j)

        bkwd[t] = backward_step(Y[t], likelihood_fn, pi, bkwd[t+1], L)

    return bkwd


#=============================================================
# forward pass of the data to make a regime assignment
#=============================================================
def forward_step(obs, likeihood_fn, pi_ztmin1, m_tplus1, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    prob = np.exp(np.log(pi_ztmin1) + likeihood_fn.logpdf(obs) + m_tplus1)
    return prob/np.sum(prob)

def state_assignments(Y, bkwd, mean_func, cov_func, likelihood, params, **kwargs):
    '''
    Sample the state assignment for z 1 to T and update the sets Y_k accordingly
    '''

    pi = params['pi']
    theta = params['theta']
    L,T = params['L'],params['T']

    # start with n_{jk} = 0
    n = np.zeros(shape=(L,L), dtype=np.int16)

    options = np.arange(0,L,dtype=np.int16)
    # starting state assignment
    # TODO: Is this correct?????
    # z_tmin1 = np.argmax(bkwd[0])
    z_tmin1 = np.random.choice(options)
    z = np.zeros(shape=T, dtype=np.int16)
    Yk = {}

    for t, yt in enumerate(Y):

        mus = [mean_func(theta, t, Y, j) for j in range(L)]
        sigmas = [cov_func(theta, t, Y, j) for j in range(L)]
        likelihood_fn = likelihood(mus, sigmas)

        # (a) compute the probability f_k(yt)
        prob_fk = forward_step(yt, likelihood_fn, pi[z_tmin1], bkwd[t], L)
        # print(prob_fk)
        # (b) sample a new z_t
        z[t] = np.random.choice(options, p=prob_fk.astype(np.float64))

        # (c) increment n
        n[z_tmin1, z[t]] += 1

        z_tmin1 = z[t]

    return {
        'z': z,
        'n': n
    }

def forward_step_slds(state, obs, likeihood_fn_state, likeihood_fn_obs, pi_ztmin1, m_tplus1, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    prob = np.exp(np.log(pi_ztmin1) + likeihood_fn_state.logpdf(state) + m_tplus1)# + likeihood_fn_obs.logpdf(obs))
    return prob/np.sum(prob)

def state_assignments_slds(Y_tilde, Y, bkwd, mean_func_state, cov_func_state, mean_func_obs, cov_func_obs, likelihood, params, **kwargs):
    '''
    Sample the state assignment for z 1 to T and update the sets Y_k accordingly
    '''

    pi = params['pi']
    theta = params['theta']
    L,T = params['L'],params['T']

    # start with n_{jk} = 0
    n = np.zeros(shape=(L,L), dtype=np.int16)

    options = np.arange(0,L,dtype=np.int16)
    # starting state assignment
    # TODO: Is this correct?????
    # z_tmin1 = np.argmax(bkwd[0])
    z_tmin1 = np.random.choice(options)
    z = np.zeros(shape=T, dtype=np.int16)
    Yk = {}

    for t, yt in enumerate(Y):

        mus = [mean_func_state(theta, t, Y_tilde, j) for j in range(L)]
        sigmas = [cov_func_state(theta, t, Y_tilde, j) for j in range(L)]
        likelihood_fn_state = likelihood(mus, sigmas)

        mus_obs = [mean_func_obs(params, t, Y_tilde, Y, j) for j in range(L)]
        sigmas_obs = [cov_func_obs(params, t, Y_tilde, Y, j) for j in range(L)]
        likelihood_fn_obs = likelihood(mus_obs, sigmas_obs)

        # (a) compute the probability f_k(yt)
        prob_fk = forward_step_slds(Y_tilde[t], yt, likelihood_fn_state, likelihood_fn_obs, pi[z_tmin1], bkwd[t], L)
        # print(prob_fk)
        # (b) sample a new z_t
        z[t] = np.random.choice(options, p=prob_fk.astype(np.float64))

        # (c) increment n
        n[z_tmin1, z[t]] += 1

        z_tmin1 = z[t]

    return {
        'z': z,
        'n': n
    }

#=============================================================
# some steps for the hdp hmm model
#=============================================================

def step_3(state_assignments, params):

    L = params['L']
    alpha = params['alpha']
    kappa = params['kappa']
    beta = params['beta']
    z = state_assignments['z']

    m = np.zeros(shape=(L,L), dtype=np.int16)

    J = [[[] for i in range(L)] for i in range(L)]

    for t, zt in enumerate(z):
        if t > 0:
            J[z[t-1]][zt].append(t)


    for j in range(1, L):
        for k in range(1, L):

            n = 0
            for i in J[j][k]:
                prob = (alpha*beta[k] + kappa)/(n + alpha*beta[k] + kappa)
                x = np.random.binomial(n=1, p=prob)
                m[j,k] += x
                n += 1

    ##############
    # STEP (b)
    ##############
    rho = kappa/(alpha + kappa)
    w = np.zeros(shape=(L,L), dtype=np.int16)
    for j in range(L):

        prob = rho/(rho + beta[j]*(1-rho))
        w[j,j] = np.random.binomial(n=m[j,j], p=prob)

    mbar = m - w

    return {
        'm': m,
        'w': w,
        'mbar': mbar
    }


def update_beta(mbar, params):
    '''
    Update the global parameter beta according to the specification above
    '''
    gamma = params['gamma']
    L = params['L']

    # TODO: which axis are you supposed to sum over here
    mbar_dot = np.sum(mbar, axis=0)
    return np.random.dirichlet(gamma/L + mbar_dot)

def update_pi(n, z, params, **kwargs):

    L = params['L']
    kappa = params['kappa']
    alpha = params['alpha']
    beta = params['beta']

    pi = np.zeros(shape=(L,L))

    for i, pik in enumerate(pi):

        # update the pis
        kappa_ = np.zeros(shape=L)
        kappa_[i] = kappa
        pi[i] = np.random.dirichlet(alpha * beta + kappa_ + n[i])

    return pi

def update_params(params, pi, beta, theta):

    params['pi'] = pi
    params['beta'] = beta
    params['theta'] = theta

    return params

#=============================================================
# put the sampler together
#=============================================================
def blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, likelihood, theta_update, priors, num_iter=100, return_assignments=False, verbose=True, **kwargs):

    params = starting_params
    Y = np.array(Y)
    assignments = np.zeros(shape=(len(Y), num_iter))

    for epic in range(num_iter):
        start = time.time()

        bkwd_messages = backward_algorithm(Y, mean_func, cov_func, likelihood, params)
        state_par = state_assignments(Y, bkwd_messages, mean_func, cov_func, likelihood, params)
        z, n = state_par['z'], state_par['n']

        step3_update = step_3(state_par, params)
        mbar = step3_update['mbar']
        beta = update_beta(mbar, params)

        pi = update_pi(n, z, params)
        theta = theta_update(Y, state_par, params, priors=priors, **kwargs)

        params = update_params(params, pi, beta, theta)

        if return_assignments:
            assignments[:,epic] = np.array(z, dtype=np.int16)

        if epic % 10 == 0:
            assert 'chains' in kwargs
            if verbose:
                seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=kwargs['chains'], seq2=z)
                print("Iteration: %i, # inf chain: %i, time: %0.2f, hamming_dist: %0.3f"%(epic, len(np.unique(z)) ,time.time() - start, hamming_val))

            else:
                print("Iteration: %i, # inf chain: %i" %(epic, len(np.unique(z))) )

    if return_assignments:
        return params, np.array(z), assignments

    return params, np.array(z)



#=============================================================
############### Specific parameter updates per model
#=============================================================
def update_theta(Y, fwd_vals, params, priors, **kwargs):

    mu0, sig0, nu, Delta = priors
    # how many time to iterate these samples?
    num_iter = 10
    L = params['L']
    theta = params['theta']
    z = fwd_vals['z']

    Yk = [Y[np.where(z == j)[0]] for j in range(L)]

    for i in range(num_iter):
        for k in range(0,L):

            ykk = Yk[k]
            sig2 = theta[k][1]**2

            # update mu
            sig0_inv = 1/sig0
            SigHat = (sig0_inv + len(ykk)/(sig2))**-1
            muHat = SigHat * (mu0*sig0_inv  + np.sum(ykk)/sig2)
            theta[k][0] = np.random.normal(loc=muHat, scale=np.sqrt(SigHat))

            # update sigma
            nu_k = nu + len(ykk)/2
            nukDeltak = Delta + np.sum(np.square(ykk - theta[k][0]))/2
            theta[k][1] = np.sqrt(stats.invgamma(a=nu_k, scale=nukDeltak).rvs())

    return theta

# sufficient statistics for the SLDS model
def slds_sufficient_statistics(Y, Y_bar, fwd_pass, ar, params, **kwargs):
    S_ybarybar, S_yybar, S_yy = [], [], []
    M, K = params['priors']['M'], params['priors']['K']

    for j in range(params['L']):
        # for each model
        indexes = np.where(fwd_pass['z'] == j)[0]

        if len(indexes) <= 0:
            y_b_k = K
            yk_yb_k = np.dot(M, K)
            y_k = np.dot(np.dot(M, K), M.T)
        else:
            Y_k = Y[indexes].T
            Y_bar_k = Y_bar[indexes].T

            y_b_k = np.dot(Y_bar_k, Y_bar_k.T) + K
            yk_yb_k = np.dot(Y_k, Y_bar_k.T) + np.dot(M, K)
            y_k = np.dot(Y_k, Y_k.T) + np.dot(np.dot(M, K), M.T)

        S_ybarybar.append( y_b_k )
        S_yybar.append( yk_yb_k )
        S_yy.append( y_k )

    return (
        S_ybarybar,
        S_yybar,
        S_yy
    )

def update_slds_theta(Y, fwd_vals, params, priors, ar=1, **kwargs):

    S_0, n_0 = priors
    L = params['L']
    D = params['D']
    theta = params['theta']
    n = fwd_vals['n']

    # set pseudo_obs = pseudo_obs
    if ar == 2:
        if D > 1:
            mx, mn = np.max(Y.shape), np.min(Y.shape)
            Y_bar = np.concatenate([np.zeros_like(Y), np.zeros_like(Y)], axis=-1)
            Y_bar[0:2,:] = np.concatenate([Y[0,:], Y[0,:]])
            Y_bar[1:,:mn] = Y[:-1]
            Y_bar[2:,mn:] = Y[:-2]
        else:
            Y_bar = np.zeros(shape=(Y.shape[0], D*2))
            Y_bar[0:2] = Y[0]
            Y_bar[1:,0] = Y[:-1]
            Y_bar[2:,1] = Y[:-2]
    else:
        Y_bar = np.zeros_like(Y)
        Y_bar[0] = Y[0]
        Y_bar[1:] = Y[0:-1]

    # step 5 caluculate the sufficient statistics for the pseudo_obs
    S_ybarybar, S_yybar, S_yy = slds_sufficient_statistics(Y, Y_bar, fwd_vals, ar, params)

    for k in range(0,L):

        S_ybarybar_k, S_yybar_k, S_yy_k = S_ybarybar[k], S_yybar[k], S_yy[k]
        S_ybarybar_k_inv = np.linalg.pinv(S_ybarybar_k)

        # sigma_k
        Sy_vert_ybar = S_yy_k - S_yybar_k.dot(S_ybarybar_k_inv).dot(S_yybar_k.T)
        sig = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[k])+n_0).rvs()

        # A = np.zeros(shape=(D,D))
        # Sigma = np.zeros(shape=(D,D))

        # for sig in sigs:
        if type(sig) != np.ndarray:
            sig = np.reshape(sig, newshape=(1,1))
        # Sigma += sig
        sig_inv = np.linalg.pinv(sig)
        M = np.dot(S_yybar_k, S_ybarybar_k_inv)
        # print(M)
        # V = sig_inv
        K_inv = S_ybarybar_k_inv
        # A += stats.matrix_normal(mean=M, rowcov=sig, colcov=K_inv).rvs()
        if ar == 2:
            A = stats.multivariate_normal(mean=M.reshape(-1,), cov=np.kron(K_inv, sig), allow_singular=True).rvs()
            # print(A)
            A = A.reshape(2,-D,D)
        else:
            A = stats.multivariate_normal(mean=M.reshape(-1,), cov=np.kron(K_inv, sig), allow_singular=True).rvs()
            A = A.reshape(D,-D)

        theta[k]['A'] = A
        theta[k]['sigma'] = sig

    return theta

def HDP_HMM_State_Sampling(Y, fwd_vals, params):

    D,T,C,R = params['D'], params['T'], params['C'], params['R']
    R_inv = np.linalg.pinv(R)
    states = np.zeros(shape=(T,C.shape[1]))

    # Initialise the first state:
    # states[0,:] = np.ones(C.shape[1])

    # algorithm 1
    # initialize filter
    Delta = np.zeros(shape=(T,C.shape[1],C.shape[1]), dtype=np.float64)
    Theta = np.zeros(shape=(T,C.shape[1]), dtype=np.float64)

    delta_tp1 = C.T.dot(R_inv).dot(C)
    Theta_tp1 = C.T.dot(R_inv).dot(Y[-1])

    Delta[-1] = C.T.dot(R_inv).dot(C)
    Theta[-1] = (C.T.dot(R_inv).dot(Y[-1])).reshape(-1,)
    #############################################################
    # Algorithm 1 - Numerically stable backwards Kalman Information filter
    #############################################################
    for tf in range(len(Y)-1):

        t = (T-2)-tf # we need the reverse index
        sig_assigned = params['theta'][fwd_vals['z'][t+1]]['sigma']
        A_assigned = params['theta'][fwd_vals['z'][t+1]]['A']
        sig_assined_inv = np.linalg.pinv(sig_assigned)

        # print(delta_tp1)
        J_tilde = Delta[t+1].dot(np.linalg.pinv((Delta[t+1] + sig_assined_inv)))
        L_tilde = np.eye(D) - J_tilde
        #
        # delta_tp1_t = A_assigned.T.dot(L_tilde.dot(Delta[t+1]).dot(L_tilde.T) + J_tilde.dot(sig_assined_inv).dot(J_tilde.T)).dot(A_assigned)
        delta_tp1_t = A_assigned.T.dot(sig_assined_inv).dot(A_assigned) - A_assigned.T.dot(sig_assined_inv).dot(np.linalg.pinv(sig_assined_inv + Delta[t+1])).dot(sig_assined_inv).dot(A_assigned)
        # Theta_tp1_t = A_assigned.T.dot(L_tilde).dot(Theta[t+1])
        Theta_tp1_t = A_assigned.T.dot(sig_assined_inv).dot(np.linalg.pinv(sig_assined_inv + Delta[t+1])).dot(Theta[t+1])

        # delta_tp1 = delta_tp1_t + C.T.dot(R_inv).dot(C)
        # Theta_tp1 = Theta_tp1_t + C.T.dot(R_inv).dot(Y[t])
        # print(t)
        Delta[t] = delta_tp1_t + C.T.dot(R_inv).dot(C)
        Theta[t] = (Theta_tp1_t.reshape(-1,) + C.T.dot(R_inv).dot(Y[t]).reshape(-1,))

    # print(Theta)
    for t,_ in enumerate(Y):
        sig_assigned = params['theta'][fwd_vals['z'][t]]['sigma']
        A_assigned = params['theta'][fwd_vals['z'][t]]['A']
        sig_assined_inv = np.linalg.pinv(sig_assigned)
        if t == 0:
            Sigma = np.linalg.pinv((sig_assined_inv + Delta[t]))
            mean = Sigma.dot(Theta[t])
        else:
            Sigma = np.linalg.pinv((sig_assined_inv + Delta[t]))
            mean = Sigma.dot(sig_assined_inv.dot(A_assigned).dot(states[t-1]) + Theta[t])

        # print(Sigma)
        states[t] = np.random.multivariate_normal(mean=mean.reshape(-1,), cov=Sigma)
    return states

#=============================================================
# Here are the different sampler models
#=============================================================

def sticky_HMM(Y, starting_params, priors = [0,200,1,10], num_iter=100, verbose=True, return_assignments=False, **kwargs):

    def mean_func(theta, t, Y, j):
        return theta[j][0]

    def cov_func(theta, t, Y, j):
        return theta[j][1]

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, stats.norm, update_theta, priors, num_iter, return_assignments, verbose, **kwargs)

def sticky_HDP_AR(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):

    def mean_func(theta, t, Y, j):
        if t == 0:
            return Y[t]*theta[j]['A'][0][0]
        return Y[t-1]*theta[j]['A'][0][0]

    def cov_func(theta, t, Y, j):
        return np.sqrt(theta[j]['sigma'][0][0])

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, stats.norm, update_slds_theta, priors, num_iter, return_assignments, verbose, **kwargs)

def sticky_Multi_HDP_AR(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):
    if starting_params['D'] == 1:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'])[0]
            return np.dot(Y[t-1], theta[j]['A'])[0]
    else:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'])
            return np.dot(Y[t-1], theta[j]['A'])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors, num_iter, return_assignments, verbose, **kwargs)


def sticky_Multi_HDP_AR2(Y, starting_params, priors, num_iter=100,  verbose=True, return_assignments=False, **kwargs):
    if starting_params['D'] == 1:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'][0])[0] + np.dot(Y[t], theta[j]['A'][1])[0]
            elif t == 1:
                return np.dot(Y[t-1], theta[j]['A'][0])[0] + np.dot(Y[t-1], theta[j]['A'][1])[0]
            return np.dot(Y[t-1], theta[j]['A'][0])[0] + np.dot(Y[t-2], theta[j]['A'][1])[0]
    else:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'][0]) + np.dot(Y[t], theta[j]['A'][1])
            elif t == 1:
                return np.dot(Y[t-1], theta[j]['A'][0]) + np.dot(Y[t-1], theta[j]['A'][1])
            return np.dot(Y[t-1], theta[j]['A'][0]) + np.dot(Y[t-2], theta[j]['A'][1])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors, num_iter, return_assignments,  verbose, ar=2, **kwargs)


def SLDS_blocked_sampler(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):

    Y = np.array(Y)
    assignments = np.zeros(shape=(len(Y), num_iter))

    if starting_params['D'] == 1:
        def mean_func_state(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'])[0]
            return np.dot(Y[t-1], theta[j]['A'])[0]
        def mean_func_obs(params, t, Y_tilde, Y, j):
            return Y_tilde[t] * params['C']
    else:
        def mean_func_state(theta, t, Y, j):
            if t == 0:
                return np.dot(Y[t], theta[j]['A'])
            return np.dot(Y[t-1], theta[j]['A'])

        def mean_func_obs(params, t, Y_tilde, Y, j):
            return np.dot(params['C'], Y_tilde[t])

    def cov_func_state(theta, t, Y, j):
        return theta[j]['sigma']

    def cov_func_obs(params, t, Y_tilde, Y, j):
        return params['R']

    params = starting_params
    Y_tilde = kwargs['Y_tilde'] if 'Y_tilde' in kwargs else np.array(Y)

    for epic in range(num_iter):
        start = time.time()

        bkwd_messages = backward_algorithm(Y_tilde, mean_func_state, cov_func_state, MultivariateNormal, params)
        # state_par = state_assignments_slds(Y_tilde, Y, bkwd_messages, mean_func_state, cov_func_state,  mean_func_obs, cov_func_obs, MultivariateNormal, params)
        state_par = state_assignments(Y_tilde, bkwd_messages, mean_func_state, cov_func_state, MultivariateNormal, params)
        z, n = state_par['z'], state_par['n']

        Y_tilde = HDP_HMM_State_Sampling(Y, state_par, params)
        exp_val = Y_tilde.dot(params['C'].T) # for the R update
        # print(exp_val)
        # print(np.sum(np.square(Y - exp_val)))

        if params['D'] == 1:
            Y_tilde = Y_tilde[:,0]
            exp_val = exp_val[:,0]

        step3_update = step_3(state_par, params)
        mbar = step3_update['mbar']

        beta = update_beta(mbar, params)

        pi = update_pi(n, z, params)
        theta = update_slds_theta(Y_tilde, state_par, params, priors=priors)

        # for the SLDS we also need to update R
        # cov = np.cov - exp_val).T).dot((Y - exp_val)) if len(Y.shape) > 1 else (Y.reshape(-1,1) - exp_val).T.dot((Y.reshape(-1,1) - exp_val))
        cov = params['T'] * (np.cov(Y - exp_val, rowvar=False) if len(Y.shape) > 1 else np.cov(Y.reshape(-1,) - exp_val.reshape(-1,)))
        R = stats.invwishart(scale=cov + params['R0'], df=params['r0'] + params['T']).rvs()

        params['R'] = np.array([[R]]) if type(R) != np.ndarray else R;
        params = update_params(params, pi, beta, theta)

        if return_assignments:
            assignments[:,epic] = np.array(z, dtype=np.int16)

        if epic % 10 == 0:
            assert 'chains' in kwargs
            if verbose:
                seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=kwargs['chains'], seq2=z)
                print("Iteration: %i, # inf chain: %i, time: %0.2f, hamming_dist: %0.3f"%(epic, len(np.unique(z)) ,time.time() - start, hamming_val))

            else:
                print("Iteration: %i, # inf chain: %i" %(epic, len(np.unique(z))) )

    if return_assignments:
        return params, np.array(z), assignments

    return params, np.array(z)
