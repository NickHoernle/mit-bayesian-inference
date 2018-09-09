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
    L = np.linalg.inv(np.array(x, dtype=np.float32))
    return L

#=============================================================
# backward pass of the data
#=============================================================
def backward_step(obs, likeihood_fn, pi, m_tplus1, L, **kwargs):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = (np.log(pi) + likeihood_fn.logpdf(obs) + m_tplus1)
    return sp.misc.logsumexp(messages, axis=1)

def backward_algorithm(Y, mean_func, cov_func, likelihood, params, **kwargs):
    '''
    Calculate the backward messages for all time T...1
    '''

    pi, theta, L, T = params['pi'], params['theta'], params['L'],params['T']

    # we have L models and T time steps
    bkwd = np.zeros(shape=(T+1, L), dtype=np.float32)

    # (a) set the messages T+1,T t
    # we are working in log space
    bkwd[-1,:] = 0

    # (b) compute the backward messages
    for tf, yt in enumerate(Y):

        # account for 0 indexing and
        # we need the reverse index
        t = (T-1)-tf

        mu_k_j = [[mean_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
        sigma_k_j = [[cov_func(theta, t, Y, j) for j in range(L)] for k in range(L)]
        likelihood_fn = likelihood(mu_k_j, sigma_k_j)

        # evaluate the backward message for each time-step and each assignment for z
        bkwd_val = backward_step(Y[t], likelihood_fn, pi, bkwd[t+1], L)
        bkwd[t] = bkwd_val - sp.misc.logsumexp(bkwd_val)

    return bkwd


#=============================================================
# forward pass of the data to make a regime assignment
#=============================================================
def forward_step(obs, likeihood_fn, pi_ztmin1, m_tplus1, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''

    prob = np.log(pi_ztmin1) + likeihood_fn.logpdf(obs) + m_tplus1
    return prob

def state_assignments(Y, bkwd, mean_func, cov_func, likelihood, params, **kwargs):
    '''
    Sample the state assignment for z 1 to T and update the sets Y_k accordingly
    '''

    pi, theta, L, T = params['pi'], params['theta'], params['L'],params['T']

    # start with n_{jk} = 0
    n = np.zeros(shape=(L,L), dtype=np.int16)
    options = np.arange(0,L,dtype=np.int16)

    z_tmin1 = np.random.choice(options)
    z = np.zeros(shape=T, dtype=np.int16)

    for t in range(0, len(Y)):

        mus = [mean_func(theta, t, Y, j) for j in range(L)]
        sigmas = [cov_func(theta, t, Y, j) for j in range(L)]
        likelihood_fn = likelihood(mus, sigmas)

        # (a) compute the probability f_k(yt)
        prob_fk = np.exp(forward_step(Y[t], likelihood_fn, pi[z_tmin1], bkwd[t], L))

        # (b) sample a new z_t
        prob_fk /= np.sum(prob_fk)
        zt = np.random.choice(options, p=prob_fk)

        # (c) increment n
        n[z_tmin1, zt] += 1
        z_tmin1 = zt
        z[t] = zt

    return {'z': z, 'n': n}

#=============================================================
# some steps for the hdp hmm model
#=============================================================

def step_3(state_assignments, params):

    L, alpha, kappa, beta = params['L'], params['alpha'], params['kappa'], params['beta']
    z = state_assignments['z']

    # initialise m to 0
    m = np.zeros(shape=(L,L), dtype=np.int16)
    J = np.zeros(shape=(L,L), dtype=np.int16)

    for t in range(1, len(z)):
        J[z[t-1], z[t]] += 1

    for j in range(0, L):
        for k in range(0, L):

            # initialise n to 0
            # n is the number of customers in the jth restaurant eating the kth dish
            for n in range(1, J[j,k]+1):
                prob = (alpha*beta[k] + kappa)/(n + alpha*beta[k] + kappa)
                x = np.random.binomial(n=1, p=prob)
                m[j,k] += x

    ##############
    # STEP (b)
    ##############
    rho = kappa/(alpha + kappa)
    w = np.zeros(shape=(L,L), dtype=np.int16)

    for j in range(L):

        prob = rho/(rho + beta[j]*(1-rho))
        w[j,j] = np.random.binomial(n=m[j,j], p=prob)

    # mbar = mjk if k != j and (m-w) if k==j
    mbar = m - w

    return {'m': m, 'w': w, 'mbar': mbar}


def update_beta(mbar, params):
    '''
    Update the global parameter beta according to the specification above
    '''
    gamma, L = params['gamma'], params['L']

    # TODO: I STILL DON'T KNOW ABOUT THIS ONE
    # mbar_dot = np.sum(mbar, axis=1)
    mbar_dot = np.sum(mbar, axis=0) # I think this one
    return np.random.dirichlet(gamma/L + mbar_dot)

def update_pi(n, z, params, **kwargs):

    L, kappa, alpha, beta = params['L'], params['kappa'], params['alpha'], params['beta']
    pi = np.zeros(shape=(L,L), dtype=np.float32)

    kappa_ = np.eye(L) * kappa
    for i, pik in enumerate(pi):

        # update the pis
        # TODO: I STILL DON'T KNOW ABOUT THIS ONE
        # pi[i] = np.random.dirichlet(alpha * beta + kappa_[i] + n[:,i]) # this one?
        pi[i] = np.random.dirichlet(alpha * beta + kappa_[i] + n[i])

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
    Y = np.array(Y, dtype=np.float32)
    assignments = np.zeros(shape=(len(Y), num_iter), dtype=np.int16)
    hamming_dist = np.zeros(shape=num_iter, dtype=np.float32)

    for epic in range(num_iter):

        start = time.time()

        bkwd_messages = backward_algorithm(Y, mean_func, cov_func, likelihood, params)
        state_par = state_assignments(Y, bkwd_messages, mean_func, cov_func, likelihood, params)

        step3_update = step_3(state_par, params)
        beta = update_beta(step3_update['mbar'], params)

        pi = update_pi(state_par['n'], state_par['z'], params)
        theta = theta_update(Y, state_par, params, priors=priors, **kwargs)

        params = update_params(params, pi, beta, theta)

        if return_assignments:
            assignments[:,epic] = np.array(state_par['z'], dtype=np.int16)

        if verbose:
            seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=kwargs['chains'], seq2=state_par['z'])
            hamming_dist[epic] = hamming_val/len(Y)
            if epic % 10 == 0:
                print("Iteration: %i, # inf chain: %i, time: %0.2f, hamming_dist: %0.3f"%(epic, len(np.unique(state_par['z'])) ,time.time() - start, hamming_val))

    if return_assignments:
        return params, np.array(state_par['z']), assignments, hamming_dist

    return params, np.array(z)



#=============================================================
############### Specific parameter updates per model
#=============================================================
def update_theta(Y, fwd_vals, params, priors, **kwargs):

    mu0, sig0, nu, Delta = priors

    # how many time to iterate these samples?
    num_iter = 1
    L, theta = params['L'], params['theta']
    z = fwd_vals['z']

    Yk = [Y[np.where(z == j)[0]] for j in range(L)]
    # print(z)
    # system.exit(0)

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

    M, K = params['priors']['M'], params['priors']['K']

    M_dot_K = np.dot(M, K)
    M_dot_K_dot_MT = np.dot(np.dot(M, K), M.T)

    S_ybarybar = np.zeros(shape=(params['L'], K.shape[0], K.shape[1]))
    S_yybar = np.zeros(shape=(params['L'], M_dot_K.shape[0], M_dot_K.shape[1]))
    S_yy = np.zeros(shape=(params['L'], M_dot_K_dot_MT.shape[0], M_dot_K_dot_MT.shape[1]))

    for j in range(params['L']):
        # for each model
        indexes = np.where(fwd_pass['z'] == j)[0]

        S_ybarybar[j] += K
        S_yybar[j] += M_dot_K
        S_yy[j] += M_dot_K_dot_MT

        if len(indexes) > 0:

            Y_k = Y[indexes]
            Y_bar_k = Y_bar[indexes]

            y_b_k = np.dot(Y_bar_k.T, Y_bar_k)
            yk_yb_k = np.dot(Y_k.T, Y_bar_k)
            y_k = np.dot(Y_k.T, Y_k)

            S_ybarybar[j] += y_b_k
            S_yybar[j] += yk_yb_k
            S_yy[j] += y_k

    return ( S_ybarybar, S_yybar, S_yy )

def update_slds_theta(Y, fwd_vals, params, priors, ar=1, **kwargs):

    S_0, n_0 = priors
    L, D, theta = params['L'], params['D'], params['theta']
    n = fwd_vals['n']

    # set pseudo_obs = pseudo_obs
    if ar == 2:
        if D > 1:
            mx, mn = np.max(Y.shape), np.min(Y.shape)
            Y_bar = np.concatenate([np.zeros_like(Y), np.zeros_like(Y)], axis=-1).astype(np.float32)
            Y_bar[0,:] += np.concatenate([Y[0,:], Y[0,:]])
            Y_bar[1:,:mn] += Y[:-1]
            Y_bar[1,mn:] += Y[0,:]
            Y_bar[2:,mn:] += Y[:-2]
        else:
            Y_bar = np.zeros(shape=(Y.shape[0], D*2), dtype=np.float32)
            Y_bar[0] += Y[0]
            Y_bar[1:,0] += Y[:-1]
            Y_bar[1,1] += Y[0]
            Y_bar[2:,1] += Y[:-2]
    else:
        # lagged observations
        Y_bar = np.zeros_like(Y, dtype=np.float32)
        Y_bar[0] += Y[0]
        Y_bar[1:] += Y[:-1]

    # step 5 caluculate the sufficient statistics for the pseudo_obs
    S_ybarybar, S_yybar, S_yy = slds_sufficient_statistics(Y, Y_bar, fwd_vals, ar, params)

    for k in range(0,L):

        S_ybarybar_k, S_yybar_k, S_yy_k = S_ybarybar[k], S_yybar[k], S_yy[k]
        S_ybarybar_k_inv = invert(S_ybarybar_k)

        # sigma_k
        Sy_vert_ybar = S_yy_k - np.dot(np.dot(S_yybar_k, S_ybarybar_k_inv), S_yybar_k.T)
        # sig = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[:,k])+n_0).rvs().astype(np.float32)
        sig = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[k])+n_0).rvs().astype(np.float32)

        if type(sig) != np.ndarray:
            sig = np.reshape(sig, newshape=(1,1))

        sigma_ = theta[k]['sigma']
        # sigma_ = sig
        sigma_inv_ = invert(sigma_)

        # TODO: I there is ambiguity in paper, not sure about this
        M = np.dot(S_yybar_k, S_ybarybar_k_inv) # PhD suggests this one
        # K_inv = S_ybarybar_k_inv

        if ar == 2:
            A = stats.matrix_normal(mean=M, rowcov=sigma_inv_, colcov=S_ybarybar_k_inv).rvs()
            A = np.array([A[:D,D:], A[:D,:D]], dtype=np.float32)
        else:
            A = stats.matrix_normal(mean=M, rowcov=sigma_inv_, colcov=S_ybarybar_k_inv).rvs().reshape(D, -D).astype(np.float32)

        theta[k]['A'] = A
        theta[k]['sigma'] = sig

    return theta

def HDP_HMM_State_Sampling(Y, fwd_vals, params):

    D,T,C,R,M = params['D'], params['T'], params['C'], params['R'], params['priors']['M']
    R_inv = invert(R)
    states = np.zeros(shape=(T,C.shape[1]), dtype=np.float32)

    # Initialise the first state:
    # states[0,:] = np.ones(C.shape[1])
    Deltab = np.zeros(shape=(T,C.shape[1],C.shape[1]), dtype=np.float32)
    Thetab = np.zeros(shape=(T,C.shape[1]), dtype=np.float32)

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
        L_tilde = np.eye(D, dtype=np.float32) - J_tilde
        #
        delta_tp1_t = A_assigned.T.dot(L_tilde.dot(Deltab[t]).dot(L_tilde.T) + J_tilde.dot(sig_assined_inv).dot(J_tilde.T)).dot(A_assigned)
        # delta_tp1_t = A_assigned.T.dot(sig_assined_inv).dot(A_assigned) - A_assigned.T.dot(sig_assined_inv).dot(np.linalg.pinv(sig_assined_inv + Delta[t+1])).dot(sig_assined_inv).dot(A_assigned)
        Theta_tp1_t = A_assigned.T.dot(L_tilde).dot(Thetab[t])
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
        sig_assigned = params['theta'][fwd_vals['z'][t]]['sigma']
        A_assigned = params['theta'][fwd_vals['z'][t]]['A']
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
        if t <= 0:
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
                return np.dot(theta[j]['A'][1], Y[t])[0] + np.dot(theta[j]['A'][0], Y[t])[0]
            elif t == 1:
                return np.dot(theta[j]['A'][1], Y[t-1])[0] + np.dot(theta[j]['A'][0], Y[t-1])[0]
            return np.dot(theta[j]['A'][1], Y[t-1])[0] + np.dot(theta[j]['A'][0], Y[t-2])[0]
    else:
        def mean_func(theta, t, Y, j):
            if t == 0:
                return np.dot(theta[j]['A'][1], Y[t]) + np.dot(theta[j]['A'][0], Y[t])
            elif t == 1:
                return np.dot(theta[j]['A'][1], Y[t-1]) + np.dot(theta[j]['A'][0], Y[t-1])
            return np.dot(theta[j]['A'][1], Y[t-1]) + np.dot(theta[j]['A'][0], Y[t-2])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors, num_iter, return_assignments,  verbose, ar=2, **kwargs)


def SLDS_blocked_sampler(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):

    Y = np.array(Y, dtype=np.float32)
    assignments = np.zeros(shape=(len(Y), num_iter))
    hamming_dist = np.zeros(shape=num_iter, dtype=np.float32)

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
        state_par = state_assignments(Y_tilde, bkwd_messages, mean_func_state, cov_func_state, MultivariateNormal, params)
        Y_tilde = HDP_HMM_State_Sampling(Y, state_par, params)

        params['states'] = Y_tilde
        z, n = state_par['z'], state_par['n']

        exp_val = Y_tilde.dot(params['C'].T) # for the R update

        if params['D'] == 1:
            Y_tilde = Y_tilde[:,0]
            exp_val = exp_val[:,0]

        step3_update = step_3(state_par, params)
        mbar = step3_update['mbar']

        beta = update_beta(mbar, params)

        pi = update_pi(n, z, params)
        theta = update_slds_theta(Y_tilde, state_par, params, priors=priors)

        # for the SLDS we also need to update R
        cov = ((Y - exp_val).T).dot((Y - exp_val)) if len(Y.shape) > 1 else (Y.reshape(-1,1) - exp_val).T.dot((Y.reshape(-1,1) - exp_val))
        R = stats.invwishart(scale=cov + params['R0'], df=params['r0'] + params['T']).rvs()

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
