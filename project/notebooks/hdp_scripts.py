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

def invert(x):
    L = np.linalg.inv(np.linalg.cholesky(np.array(x, dtype=np.float64)))
    return np.dot(L.T,L)

#=============================================================
# backward pass of the data
#=============================================================
def backward_step(obs, likeihood_fn, pi, m_tplus1, L, **kwargs):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = np.zeros((L,L), dtype=np.float64)
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
    bkwd = np.zeros(shape=(T+1, L), dtype=np.float64)
    # (a) set the messages T+1,T t
    bkwd[-1,:] = 0

    # (b) compute the backward messages
    for tf, yt in enumerate(Y):

        t = (T-1)-tf # we need the reverse index

        mu_k_j = [[mean_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
        sigma_k_j = [[cov_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
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
    prob =  np.zeros((L), dtype=np.float64)
    prob += np.log(pi_ztmin1) + likeihood_fn.logpdf(obs) + m_tplus1
    return prob - sp.misc.logsumexp(prob) # probabilities sum to 0

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
    mus = [mean_func(theta, 0, Y, j) for j in range(L)]
    sigmas = [cov_func(theta, 0, Y, j) for j in range(L)]
    likelihood_fn = likelihood(mus, sigmas)
    p_tmin_1 = np.exp(bkwd[1] + likelihood_fn.logpdf(Y[0]))

    z_tmin1 = np.random.choice(options, p=(p_tmin_1/np.sum(p_tmin_1)).astype(np.float64))

    z = np.zeros(shape=T, dtype=np.int16)
    Yk = {}

    for t in range(1, len(Y)):

        mus = [mean_func(theta, t, Y, j) for j in range(L)]
        sigmas = [cov_func(theta, t, Y, j) for j in range(L)]
        likelihood_fn = likelihood(mus, sigmas)

        # (a) compute the probability f_k(yt)
        prob_fk = forward_step(Y[t], likelihood_fn, pi[z[t-1]], bkwd[t+1], L)

        # (b) sample a new z_t
        z[t] = np.random.choice(options, p=np.exp(prob_fk))

        # (c) increment n
        n[z_tmin1, z[t]] += 1

        z_tmin1 = z[t]

    return {'z': z, 'n': n}

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

    return {'m': m, 'w': w, 'mbar': mbar}


def update_beta(mbar, params):
    '''
    Update the global parameter beta according to the specification above
    '''
    gamma = params['gamma']
    L = params['L']

    mbar_dot = np.sum(mbar, axis=1)
    return np.random.dirichlet(gamma/L + mbar_dot)

def update_pi(n, z, params, **kwargs):

    L = params['L']
    kappa = params['kappa']
    alpha = params['alpha']
    beta = params['beta']

    pi = np.zeros(shape=(L,L), dtype=np.float64)

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
    hamming_dist = np.zeros(shape=num_iter, dtype=np.float64)

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

        if verbose:
            seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=kwargs['chains'], seq2=z)
            hamming_dist[epic] = hamming_val/len(Y)
            if epic % 10 == 0:
                print("Iteration: %i, # inf chain: %i, time: %0.2f, hamming_dist: %0.3f"%(epic, len(np.unique(z)) ,time.time() - start, hamming_val))

    if return_assignments:
        return params, np.array(z), assignments, hamming_dist

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
            Y_bar = np.concatenate([np.zeros_like(Y), np.zeros_like(Y)], axis=-1).astype(np.float64)
            Y_bar[0:2,:] += np.concatenate([Y[0,:], Y[0,:]])
            Y_bar[1:,:mn] += Y[:-1]
            Y_bar[2:,mn:] += Y[:-2]
        else:
            Y_bar = np.zeros(shape=(Y.shape[0], D*2), dtype=np.float64)
            Y_bar[0:2] += Y[0]
            Y_bar[1:,0] += Y[:-1]
            Y_bar[2:,1] += Y[:-2]
    else:
        Y_bar = np.zeros_like(Y, dtype=np.float64)
        Y_bar[0] += Y[0]
        Y_bar[1:] += Y[0:-1]

    # step 5 caluculate the sufficient statistics for the pseudo_obs
    S_ybarybar, S_yybar, S_yy = slds_sufficient_statistics(Y, Y_bar, fwd_vals, ar, params)

    for k in range(0,L):

        S_ybarybar_k, S_yybar_k, S_yy_k = S_ybarybar[k], S_yybar[k], S_yy[k]
        S_ybarybar_k_inv = invert(S_ybarybar_k)

        # sigma_k
        Sy_vert_ybar = S_yy_k - S_yybar_k.dot(S_ybarybar_k_inv).dot(S_yybar_k.T)
        sig = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[k])+n_0).rvs().astype(np.float64)

        # A = np.zeros(shape=(D,D))
        # Sigma = np.zeros(shape=(D,D))

        # for sig in sigs:
        if type(sig) != np.ndarray:
            sig = np.reshape(sig, newshape=(1,1))

        sig_inv = invert(sig)
        M = np.dot(S_yybar_k, S_ybarybar_k_inv)
        K_inv = S_ybarybar_k_inv

        if ar == 2:
            A_ = np.random.multivariate_normal(mean=M.T.reshape(-1,), cov=np.kron(K_inv, sig_inv)).astype(np.float64)
            A = np.array([a.T for a in A_.reshape(2, D, -D)], dtype=np.float64)

        else:
            A_ = np.random.multivariate_normal(mean=M.T.reshape(-1,), cov=np.kron(K_inv, sig_inv)).astype(np.float64)
            A = A_.reshape(D, -D).T

        theta[k]['A'] = A
        theta[k]['sigma'] = sig

    return theta

def HDP_HMM_State_Sampling(Y, fwd_vals, params):

    D,T,C,R,M = params['D'], params['T'], params['C'], params['R'], params['priors']['M']
    R_inv = invert(R)
    states = np.zeros(shape=(T,C.shape[1]), dtype=np.float64)

    # Initialise the first state:
    # states[0,:] = np.ones(C.shape[1])
    Deltab = np.zeros(shape=(T,C.shape[1],C.shape[1]), dtype=np.float64)
    Thetab = np.zeros(shape=(T,C.shape[1]), dtype=np.float64)

    Deltab[-1] = C.T.dot(R_inv).dot(C)
    Thetab[-1] = (C.T.dot(R_inv).dot(Y[-1].T)).reshape(-1,)

    #############################################################
    # Algorithm 1 - Numerically stable backwards Kalman Information filter
    #############################################################
    for tf in range(len(Y)-1):
        t = (T-2)-tf # we need the reverse index
        sig_assigned = params['theta'][fwd_vals['z'][t+1]]['sigma']
        A_assigned = params['theta'][fwd_vals['z'][t+1]]['A']
        sig_assined_inv = invert(sig_assigned)

        # print(delta_tp1)
        J_tilde = Deltab[t+1].dot(invert((Deltab[t+1] + sig_assined_inv)))
        L_tilde = np.eye(D, dtype=np.float64) - J_tilde
        #
        delta_tp1_t = A_assigned.T.dot(L_tilde.dot(Deltab[t+1]).dot(L_tilde.T) + J_tilde.dot(sig_assined_inv).dot(J_tilde.T)).dot(A_assigned)
        # delta_tp1_t = A_assigned.T.dot(sig_assined_inv).dot(A_assigned) - A_assigned.T.dot(sig_assined_inv).dot(np.linalg.pinv(sig_assined_inv + Delta[t+1])).dot(sig_assined_inv).dot(A_assigned)
        Theta_tp1_t = A_assigned.T.dot(L_tilde).dot(Thetab[t+1])
        # Theta_tp1_t = A_assigned.T.dot(sig_assined_inv).dot(np.linalg.pinv(sig_assined_inv + Delta[t+1])).dot(Theta[t+1])

        # delta_tp1 = delta_tp1_t + C.T.dot(R_inv).dot(C)
        # Theta_tp1 = Theta_tp1_t + C.T.dot(R_inv).dot(Y[t])
        # print(t)
        Deltab[t] = delta_tp1_t + C.T.dot(R_inv).dot(C)
        Thetab[t] = (Theta_tp1_t.reshape(-1,) + C.T.dot(R_inv).dot(Y[t]).reshape(-1,))

    # print(Deltab)
    sig_assigned = params['theta'][fwd_vals['z'][0]]['sigma']
    sig_assined_inv = invert(sig_assigned)
    Sigma = invert(sig_assined_inv + Deltab[0])
    mean = Sigma.dot(Thetab[0])
    states[0] = np.random.multivariate_normal(mean=mean.reshape(-1,), cov=Sigma)

    for t in range(1, len(Y)):
        sig_assigned = params['theta'][fwd_vals['z'][t-1]]['sigma']
        A_assigned = params['theta'][fwd_vals['z'][t-1]]['A']
        sig_assined_inv = invert(sig_assigned)

        Sigma = invert(sig_assined_inv + Deltab[t])
        mean = Sigma.dot(sig_assined_inv.dot(A_assigned).dot(states[t-1]) + Thetab[t])
        states[t] = np.random.multivariate_normal(mean=mean.reshape(-1,), cov=Sigma)
        # states[t] = mean
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
                return np.dot(theta[j]['A'], Y[t])[0]
            return np.dot(theta[j]['A'], Y[t-1])[0]
    else:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'], Y[t])
            return np.dot(theta[j]['A'], Y[t-1])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors, num_iter, return_assignments, verbose, **kwargs)


def sticky_Multi_HDP_AR2(Y, starting_params, priors, num_iter=100,  verbose=True, return_assignments=False, **kwargs):
    if starting_params['D'] == 1:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'][0], Y[t])[0] + np.dot(theta[j]['A'][1], Y[t])[0]
            elif t == 1:
                return np.dot(theta[j]['A'][0], Y[t-1])[0] + np.dot(theta[j]['A'][1], Y[t-1])[0]
            return np.dot(theta[j]['A'][0], Y[t-1])[0] + np.dot(theta[j]['A'][1], Y[t-2])[0]
    else:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'][0], Y[t]) + np.dot(theta[j]['A'][1], Y[t])
            elif t == 1:
                return np.dot(theta[j]['A'][0], Y[t-1]) + np.dot(theta[j]['A'][1], Y[t-1])
            return np.dot(theta[j]['A'][0], Y[t-1]) + np.dot(theta[j]['A'][1], Y[t-2])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors, num_iter, return_assignments,  verbose, ar=2, **kwargs)


def SLDS_blocked_sampler(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):

    Y = np.array(Y, dtype=np.float64)
    assignments = np.zeros(shape=(len(Y), num_iter))
    hamming_dist = np.zeros(shape=num_iter, dtype=np.float64)

    if starting_params['D'] == 1:
        def mean_func_state(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'], Y[t])[0]
            return np.dot(theta[j]['A'], Y[t-1])[0]

    else:
        def mean_func_state(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'], Y[t])
            return np.dot(theta[j]['A'], Y[t-1])

    def cov_func_state(theta, t, Y, j):
        return theta[j]['sigma']

    params = starting_params
    Y_tilde = kwargs['Y_tilde'] if 'Y_tilde' in kwargs else np.array(Y)
    # Y_tilde = HDP_HMM_State_Sampling(Y, {'z': kwargs['chains']}, params)
    for epic in range(num_iter):
        start = time.time()

        bkwd_messages = backward_algorithm(Y_tilde, mean_func_state, cov_func_state, MultivariateNormal, params)
        # state_par = state_assignments_slds(Y_tilde, Y, bkwd_messages, mean_func_state, cov_func_state,  mean_func_obs, cov_func_obs, MultivariateNormal, params)
        state_par = state_assignments(Y_tilde, bkwd_messages, mean_func_state, cov_func_state, MultivariateNormal, params)
        Y_tilde = HDP_HMM_State_Sampling(Y, state_par, params)
        # Y_tilde[0]
        params['states'] = Y_tilde
        # Y_tilde = HDP_HMM_State_Sampling(Y, {'z': kwargs['chains']}, params)
        # bkwd_messages = backward_algorithm(Y_tilde, mean_func_state, cov_func_state, MultivariateNormal, params)
        # state_par = state_assignments(Y_tilde, bkwd_messages, mean_func_state, cov_func_state, MultivariateNormal, params)

        z, n = state_par['z'], state_par['n']
        # z = kwargs['chains']
        exp_val = Y_tilde.dot(params['C'].T) # for the R update
        # print(exp_val.shape)
        # print(exp_val[:,0])
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
        # if R < .01:
        #     R = 0.01
        params['R'] = np.array([[R]]) if type(R) != np.ndarray else R;
        params = update_params(params, pi, beta, theta)

        if return_assignments:
            assignments[:,epic] = np.array(z, dtype=np.int16)

        # if epic % 10 == 0:
        assert 'chains' in kwargs
        if verbose:
            seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=kwargs['chains'], seq2=z)
            hamming_dist[epic] = hamming_val/len(Y)
            if epic % 10 == 0:
                print("Iteration: %i, # inf chain: %i, time: %0.2f, hamming_dist: %0.3f"%(epic, len(np.unique(z)) ,time.time() - start, hamming_val))

    if return_assignments:
        return params, np.array(z), assignments, hamming_dist

    return params, np.array(z)
