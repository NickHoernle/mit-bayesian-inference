import scipy.stats as stats
import scipy as sp
import pandas as pd
import numpy as np

#=============================================================
# backward pass of the data
#=============================================================
def backward_step(obs, likeihood_fn, pi, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = np.zeros((L,L), dtype=np.float128)
    messages += (np.log(pi) + likeihood_fn([[theta[j][0] for j in range(L)] for k in range(L)],
                                    [[theta[j][1] for j in range(L)] for k in range(L)]).logpdf(obs) + m_tplus1)
    return sp.misc.logsumexp(messages, axis=1)

def backward_step_slds(yt, ytmin1, likeihood_fn, pi, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = np.zeros((L,L), dtype=np.float128)

    mu_k_j = [[ytmin1*theta[j]['A'] for j in range(L)] for k in range(L)]
    sigma_k_j = [[theta[j]['sigma'] for j in range(L)] for k in range(L)]

    messages += (np.log(pi) + likeihood_fn(mu_k_j, sigma_k_j).logpdf(yt) + m_tplus1) # backward messages

    return sp.misc.logsumexp(messages, axis=1)

def backward_algorithm_slds(Y, params, **kwargs):
    '''
    Calculate the backward messages for all time T...1 for the SLDS.
    Here, the regime indexes a mode specific parameters that gives the
    likelihood of t from t+1.
    '''

    # pi is the transition probability
    pi = params['pi']
    # theta now indexes the transition matrix A, and the noise Sigma
    theta = params['theta']
    # regime number cutoff and length of time series respectively
    L,T = params['L'],params['T']

    # TODO: This is what changes to MVN when we step up our game to the
    # multivariate case
    likelihood_fn = stats.norm

    # we have L models and T time steps
    bkwd = np.zeros(shape=(T+1, L), dtype=np.float128)
    # (a) set the messages T+1,T = 1
    bkwd[-1,:] = 1

    # (b) compute the backward messages
    for tf, yt in enumerate(Y):

        t = (T-1)-tf # we need the reverse index
        if t == 0:
            bkwd[t] = backward_step_slds(Y[t], Y[t], likelihood_fn, pi, bkwd[t+1], theta, L)
        else:
            bkwd[t] = backward_step_slds(Y[t], Y[t-1], likelihood_fn, pi, bkwd[t+1], theta, L)

    return bkwd

def backward_algorithm(Y, params, **kwargs):
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
        bkwd[t] = backward_step(Y[t], stats.norm, pi, bkwd[t+1], theta, L)

    return bkwd


#=============================================================
# forward pass of the data to make a regime assignment
#=============================================================

def forward_step_slds(yt, ytmin1, likeihood_fn, pi_ztmin1, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    mus = [ytmin1*theta[j]['A'] for j in range(L)]
    simgas = [theta[j]['sigma'] for j in range(L)]

    prob = np.exp(np.log(pi_ztmin1) +
                  likeihood_fn(mus, simgas).logpdf(yt) +
                  m_tplus1)

    return prob/np.sum(prob)

def forward_step(obs, likeihood_fn, pi_ztmin1, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    mus = [theta[j][0] for j in range(L)]
    sigmas = [theta[j][1] for j in range(L)]
    prob = np.exp(np.log(pi_ztmin1) +
                  likeihood_fn(mus, sigmas).logpdf(obs) +
                  m_tplus1)
    return prob

def state_assignments_slds(Y, bkwd, params, **kwargs):
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

    # TODO: higher dimensions
    likelihood = stats.norm

    for t, yt in enumerate(Y):

        prob_fk = np.zeros(shape=L, dtype=np.float128)

        if t == 0:
            prob_fk += forward_step_slds(yt, yt, likelihood, pi[z_tmin1], bkwd[t], theta, L)
        else:
            # (a) compute the probability f_k(yt)
            prob_fk += forward_step_slds(yt, Y[t-1], likelihood, pi[z_tmin1], bkwd[t], theta, L)

        # (b) sample a new z_t
        z[t] = np.random.choice(options, p=prob_fk.astype(np.float64))

        # (c) increment n
        n[z_tmin1, z[t]] += 1

        z_tmin1 = z[t]

    return {
        'z': z,
        'Yk': Yk,
        'n': n
    }

def state_assignments(Y, bkwd, params, **kwargs):
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

        prob_fk = np.zeros(shape=L, dtype=np.float128)
        # (a) compute the probability f_k(yt)
        prob_fk += forward_step(yt, stats.norm, pi[z_tmin1], bkwd[t], theta, L)

        # (b) sample a new z_t
        # normalize prob_fk (TODO is this correct?????)
        prob_fk_ = np.array(prob_fk/np.sum(prob_fk), dtype=np.float64)
        z[t] = np.random.choice(options, p=prob_fk_)

        # (c) increment n
        n[z_tmin1, z[t]] += 1

        # cache y_t (not yet sure of what the eff this is for)
        if z[t] in Yk:
            Yk[z[t]].append(yt)
        else:
            Yk[z[t]] = [yt]

        z_tmin1 = z[t]

    return {
        'z': z,
        'Yk': Yk,
        'n': n
    }


#=============================================================
# sufficient statistics for the SLDS model
#=============================================================
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
            Y_k = Y[indexes]
            Y_bar_k = Y_bar[indexes]

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


def update_theta(theta, Yk, mu0, sig0, nu, Delta, params, **kwargs):

    # how many time to iterate these samples?
    num_iter = 10
    L = params['L']

    for i in range(num_iter):
        for k in range(0,L):

            ykk = np.array(Yk[k]) if k in Yk else np.array([])
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

def update_slds_theta(n, S_ybarybar, S_yybar, S_yy, theta, params, priors):

    S_0, n_0 = priors

    L = params['L']

    for k in range(0,L):

        S_ybarybar_k, S_yybar_k, S_yy_k = S_ybarybar[k], S_yybar[k], S_yy[k]

        S_ybarybar_k_inv = np.linalg.pinv(S_ybarybar_k)

        # sigma_k
        # print(S_yybar_k + S_0)
        Sy_vert_ybar = S_yy_k - S_yybar_k.dot(S_ybarybar_k_inv).dot(S_yybar_k.T)
        sig = stats.invwishart(scale=Sy_vert_ybar+S_0, df=np.sum(n[k])+n_0).rvs()
        # sig = np.square(theta[k]['sigma'])

        if type(sig) != np.ndarray:
            sig = np.reshape(sig, newshape=(1,1))

        sig_inv = np.linalg.pinv(sig)
        M = np.dot(S_yy_k, S_ybarybar_k_inv) # TODO: or is this be
        # M = np.dot(S_yybar_k, S_ybarybar_k_inv)
        # M = np.dot(Sy_vert_ybar, S_ybarybar_k_inv)
        V = sig_inv
        K_inv = S_ybarybar_k_inv
        A = stats.matrix_normal(M, rowcov=V, colcov=K_inv).rvs()

        theta[k]['A'] = A[0,0]
        theta[k]['sigma'] = np.sqrt(sig[0,0])
    # theta = [{'A': 0.9, 'sigma': np.sqrt(10)}, {'A': 0.99, 'sigma': 1}]
    return theta

def update_pi_and_theta(Yk, n, params, **kwargs):

    mu0, sig0, nu, Delta = kwargs['priors']
    theta = params['theta']

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

    theta = update_theta(theta, Yk, mu0, sig0, nu, Delta, params)

    return {
        'pi': pi,
        'theta': theta
    }

#=============================================================
# some specific updates for the slds model
#=============================================================
def update_slds_regime_params(n, S_ybarybar, S_yybar, S_yy, params, priors, **kwargs):

    theta = params['theta']

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

    theta = update_slds_theta(n, S_ybarybar, S_yybar, S_yy, theta, params, priors)

    return {
        'pi': pi,
        'theta': theta
    }



def update_params(params, pi, beta, theta):

    params['pi'] = pi
    params['beta'] = beta
    params['theta'] = theta

    return params


#=============================================================
# put it all together for the hdp hmm
#=============================================================
def blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, priors = [0,200,1,10]):

    params = starting_params

    bkwd_messages = backward_algorithm(Y, params)
    state_par = state_assignments(Y, bkwd_messages, params)
    z, Yk, n = state_par['z'], state_par['Yk'], state_par['n']

    step3_update = step_3(state_par, params)
    mbar = step3_update['mbar']
    beta = update_beta(mbar, params)
    pi_theta = update_pi_and_theta(Yk, n, params, priors=priors)

    pi = pi_theta['pi']
    theta = pi_theta['theta']

    params = update_params(params, pi, beta, theta)

    return params, np.array(z)


#=============================================================
# put it all together for the HDP-AR model
#=============================================================
def Gibbs_for_HDP_AR(Y, starting_params, priors):

    # step 1
    params = starting_params

    # step 2
    bkwd_messages = backward_algorithm_slds(Y, params)
    state_par = state_assignments_slds(Y, bkwd_messages, params)
    z, Yk, n = state_par['z'], state_par['Yk'], state_par['n']

    # set pseudo_obs = pseudo_obs
    Y_bar = np.zeros_like(Y)
    Y_bar[0] = Y[0]
    Y_bar[1:] = Y[0:-1]

    # step 5 caluculate the sufficient statistics for the pseudo_obs
    S_ybarybar, S_yybar, S_yy = slds_sufficient_statistics(Y, Y_bar, state_par, params)

    step3_update = step_3(state_par, params)
    mbar = step3_update['mbar']
    beta = update_beta(mbar, params)

    pi_theta = update_slds_regime_params(n, S_ybarybar, S_yybar, S_yy, params, priors=priors)

    pi = pi_theta['pi']
    theta = pi_theta['theta']

    params = update_params(params, pi, beta, theta)

    return params, np.array(z)
