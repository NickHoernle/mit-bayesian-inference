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
    z_tmin1 = np.argmax(bkwd[0])
    z = np.zeros(shape=T, dtype=np.int16)
    Yk = {}

    for t, yt in enumerate(Y):

        mus = [mean_func(theta, t, Y, j) for j in range(L)]
        sigmas = [cov_func(theta, t, Y, j) for j in range(L)]
        likelihood_fn = likelihood(mus, sigmas)

        # (a) compute the probability f_k(yt)
        prob_fk = forward_step(yt, likelihood_fn, pi[z_tmin1], bkwd[t], L)

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
def blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, likelihood, theta_update, priors):

    params = starting_params
    Y = np.array(Y)

    bkwd_messages = backward_algorithm(Y, mean_func, cov_func, likelihood, params)
    state_par = state_assignments(Y, bkwd_messages, mean_func, cov_func, likelihood, params)
    z, n = state_par['z'], state_par['n']

    step3_update = step_3(state_par, params)
    mbar = step3_update['mbar']
    beta = update_beta(mbar, params)

    pi = update_pi(n, z, params)
    theta = theta_update(Y, state_par, params, priors=priors)

    params = update_params(params, pi, beta, theta)

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
def slds_sufficient_statistics(Y, Y_bar, fwd_pass, params, **kwargs):
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

def update_slds_theta(Y, fwd_vals, params, priors):

    S_0, n_0 = priors
    num_iter = 10
    L = params['L']
    D = params['D']
    theta = params['theta']
    n = fwd_vals['n']

    # set pseudo_obs = pseudo_obs
    Y_bar = np.zeros_like(Y)
    Y_bar[0] = Y[0]
    Y_bar[1:] = Y[0:-1]

    # step 5 caluculate the sufficient statistics for the pseudo_obs
    S_ybarybar, S_yybar, S_yy = slds_sufficient_statistics(Y, Y_bar, fwd_vals, params)

    for k in range(0,L):

        S_ybarybar_k, S_yybar_k, S_yy_k = S_ybarybar[k], S_yybar[k], S_yy[k]
        S_ybarybar_k_inv = np.linalg.pinv(S_ybarybar_k)

        # sigma_k
        Sy_vert_ybar = S_yy_k - S_yybar_k.dot(S_ybarybar_k_inv).dot(S_yybar_k.T)

        sigs = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[k])+n_0).rvs(size=num_iter)
        A = np.zeros(shape=(D,D))
        Sigma = np.zeros(shape=(D,D))

        for sig in sigs:
            if type(sig) != np.ndarray:
                sig = np.reshape(sig, newshape=(1,1))
            Sigma += sig
            sig_inv = np.linalg.pinv(sig)
            M = np.dot(S_yybar_k, S_ybarybar_k_inv)
            V = sig_inv
            K_inv = S_ybarybar_k_inv
            A += stats.multivariate_normal(mean=M.reshape(-1,), cov=np.kron(K_inv, sig), allow_singular=True).rvs().reshape(-D,D)

        theta[k]['A'] = A/num_iter
        theta[k]['sigma'] = Sigma/num_iter

    return theta



#=============================================================
# Here are the different sampler models
#=============================================================

def sticky_HMM(Y, starting_params, priors = [0,200,1,10]):

    def mean_func(theta, t, Y, j):
        return theta[j][0]

    def cov_func(theta, t, Y, j):
        return theta[j][1]

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, stats.norm, update_theta, priors)

def sticky_HDP_AR(Y, starting_params, priors):

    def mean_func(theta, t, Y, j):
        if t == 0:
            return Y[t]*theta[j]['A'][0][0]
        return Y[t-1]*theta[j]['A'][0][0]

    def cov_func(theta, t, Y, j):
        return np.sqrt(theta[j]['sigma'][0][0])

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, stats.norm, update_slds_theta, priors)

class MultivariateNormal:
    def __init__(self, means, covariances):
        self.means = np.array(means, dtype=np.float128)
        self.cov = np.array(covariances, dtype=np.float128)
        # self.const = -0.5*np.log(np.linalg.det(2*np.pi*np.array(covariances)))
        self.cov_inv = np.linalg.pinv(self.cov.astype(np.float64)).astype(np.float128)

    def logpdf(self, x):
        mean_shape = self.means.shape
        if len(mean_shape) == 1:
            return -0.5*np.log([2*np.pi*np.linalg.det(self.cov[i].astype(np.float64)).astype(np.float128) for i,row in enumerate(self.cov)]) - 0.5*np.array((x-self.means).T.dot(self.cov_inv).dot(x-self.means))
        elif (mean_shape[0] == mean_shape[1]) and (len(mean_shape) > 2):
            return -0.5*np.log([[2*np.pi*np.linalg.det(self.cov[i,j].astype(np.float64)).astype(np.float128) for j,_ in enumerate(row)] for i,row in enumerate(self.cov)]) - 0.5*np.array([[(x-self.means[i][j]).T.dot(self.cov_inv[i][j]).dot(x-self.means[i][j]) for j,_ in enumerate(row)] for i,row in enumerate(self.means)])
        return -0.5*np.log(2*np.pi*np.linalg.det(self.cov.astype(np.float64))).astype(np.float128) - 0.5*np.array([(x-self.means[i]).T.dot(self.cov_inv[i]).dot(x-self.means[i]) for i,_ in enumerate(self.means)])

def sticky_Multi_HDP_AR(Y, starting_params, priors):

    def mean_func(theta, t, Y, j):
        if t == 0:
            return np.dot(theta[j]['A'], Y[t])
        return np.dot(theta[j]['A'], Y[t-1])

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_slds_theta, priors)
