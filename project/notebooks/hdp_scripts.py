import scipy.stats as stats
import pandas as pd
import numpy as np

def backward_step(obs, likeihood_fn, pi, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    messages = np.zeros((L,L), dtype=np.float128)
    messages += np.exp(np.log(pi) + 
                       likeihood_fn([[theta[j][0] for j in range(L)] for k in range(L)],
                                    [[theta[j][1] for j in range(L)] for k in range(L)]).logpdf(obs) +
                       np.log(m_tplus1))
    return np.sum(messages, axis=1)
    
    
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
    bkwd[-1,:] = 1
    
    # (b) compute the backward messages
    for tf, yt in enumerate(Y):
        
        t = (T-1)-tf # we need the reverse index
        bkwd[t] = backward_step(yt, stats.norm, pi, bkwd[t+1], theta, L)
    
    return bkwd
    
    
def forward_step(obs, likeihood_fn, pi_ztmin1, m_tplus1, theta, L):
    '''
    The backward message that is passed from zt to zt-1 given by the HMM:
        P(y1:T \mid z_{t-1}, \pi, \theta)
    '''
    
    prob = np.exp(np.log(pi_ztmin1) + 
                  likeihood_fn([theta[j][0] for j in range(L)], [theta[j][1] for j in range(L)]).logpdf(obs) +
                  np.log(m_tplus1))
    return prob

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
    
def step_3(state_assignments, params):
    L = params['L']
    alpha = params['alpha']
    kappa = params['kappa']
    beta = params['beta']
    z = state_assignments['z']

    m = np.zeros(shape=(L,L), dtype=np.int16)

    J = [[[] for i in range(L)] for i in range(L)]

    for t, zt in enumerate(z):
        if t == 0:
            None
        else:
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
    
    # TODO check this
    mbar_dot = np.sum(mbar, axis=1)
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
    
    
    
def update_params(params, pi, beta, theta):
    
    params['pi'] = pi
    params['beta'] = beta
    params['theta'] = theta
    
    return params
    
    
    
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