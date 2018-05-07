#!/usr/bin/env python
'''
Script to dispatch the experiments for 6.882. Run them on Odyssey and run them in parallel
'''
from multiprocessing import Pool
import os, traceback, re, sys
import copy
import numpy as np
import scipy.stats as stats
import data_generation as gen
import hdp_scripts as hdp

def generateTransitionMatrix(dim):
    A = stats.matrix_normal(mean=np.zeros(shape=(dim, dim)), rowcov=0.25, colcov=0.25).rvs()  #+ 40*np.eye(dim) + 40*waterfall
    return A

def generateCovMatrix(dim, M, scale, reg):
    noise_cov = scale*stats.invwishart(scale=10*np.eye(3), df=10).rvs()
    return noise_cov

def run_experiment_ar(iter_number, data_choice=0, **kwargs):

    np.random.seed(seed=None)

    # set the starting parameters for each model and run the models
    L, D = 10, 3

    ####################################################################
    # generate the data
    ####################################################################
    if data_choice == 0:
        chain_params = [{"A": generateTransitionMatrix(D), "sigma": generateCovMatrix(3, np.zeros(shape=(D, D)),1 ,5)},
                        {"A": generateTransitionMatrix(D), "sigma": generateCovMatrix(3, np.zeros(shape=(D, D)),2 ,5)},
                        {"A": generateTransitionMatrix(D), "sigma": generateCovMatrix(3, np.zeros(shape=(D, D)),5 ,5)}]

        res_ = gen.generate_data_nD(dim=3,
                                   num_chains=len(chain_params),
                                   length=kwargs['chain_length'] if 'chain_length' in kwargs else 1000,
                                   switch_prob=0.02,
                                   chain_params=chain_params)
        chains, Y = res_['chains'], res_['Y']

    elif data_choice == 1:
        D = 1
        chain_params = [{"A": [-0.4, -0.2], "sigma": 2}, {"A": [0.7, 0.2], "sigma": 2}, {"A": [-0.6, 0.2], "sigma": 4}]
        res_ = gen.generate_data_1D(num_chains=len(chain_params),
                                   length=kwargs['chain_length'] if 'chain_length' in kwargs else 1000,
                                   switch_prob=0.02,
                                   chain_params=chain_params,
                                   ar=2)
        chains, Y = res_['chains'], res_['Y'][:,0]

    elif data_choice == 2:
        chain_params = [{"A": generateTransitionMatrix(dim), "sigma": generateCovMatrix(3, np.zeros(shape=(dim, dim)), 4, 1)},
                        {"A": generateTransitionMatrix(dim), "sigma": generateCovMatrix(3, np.zeros(shape=(dim, dim)), 1, 1)},
                        {"A": generateTransitionMatrix(dim), "sigma": generateCovMatrix(3, np.zeros(shape=(dim, dim)), 2, 6)}]

        R = generateCovMatrix(D, 0, 2, 5)
        res_ = gen.generate_data_slsd(dim=3,
                                     R=R,
                                     num_chains=len(chain_params),
                                     length=kwargs['chain_length'] if 'chain_length' in kwargs else 1000,
                                     switch_prob=0.02,
                                     chain_params=chain_params)
        chains, Y = res_['chains'], res_['Y']

    starting_params = {}
    starting_params['pi']    = np.random.dirichlet(alpha=np.ones(L), size=L)
    starting_params['R']     = kwargs['R'] if 'R' in kwargs else 1*np.eye(D)
    starting_params['R0']    = kwargs['R0'] if 'R0' in kwargs else 1*np.eye(D)
    starting_params['r0']    = kwargs['r0'] if 'r0' in kwargs else 1e1
    starting_params['C']     = kwargs['C'] if 'C' in kwargs else np.eye(D)
    starting_params['D']     = D
    starting_params['theta'] = [{'A': np.eye(D, dtype=np.float64)*np.random.normal(0,1,size=D), 'sigma': 1*np.eye(D, dtype=np.float64)} for i in range(L)]
    starting_params['L']     = L
    starting_params['T']     = kwargs['chain_length'] if 'chain_length' in kwargs else 1000
    starting_params['alpha'] = kwargs['alpha'] if 'alpha' in kwargs else 1
    starting_params['beta']  = np.random.dirichlet(np.ones(starting_params['L']))
    starting_params['kappa'] = kwargs['kappa'] if 'kappa' in kwargs else 10
    starting_params['gamma'] = kwargs['gamma'] if 'gamma' in kwargs else 1
    starting_params['Y'] = Y
    ####################################################################
    # run the models
    ####################################################################
    if ('exec_model' not in kwargs) or (('exec_model') in kwargs and 0 in kwargs['exec_model']):
        print('RUNNING AR(1) MODEL')
        # run the AR1 model
        starting_params['priors'] = {
            'M': np.zeros(shape=(D,D), dtype=np.float64),
            'K': 10*np.eye(D, dtype=np.float64)
        }

        if data_choice == 1:
            starting_params['theta'] = [{'A': np.array([[np.random.normal(0,1)]]), 'sigma': np.array([[1]], dtype=np.float64)} for i in range(L)]
            _, res, assignments_ar1, hamming_ar1 = hdp.sticky_HDP_AR(Y, copy.deepcopy(starting_params),
                                     priors=[1*np.eye(D), D],
                                     num_iter=kwargs['num_iter'] if 'num_iter' in kwargs else 5000,
                                     return_assignments=True,
                                     verbose=True,
                                     chains=chains)
        else:
            _, res, assignments_ar1, hamming_ar1 = hdp.sticky_Multi_HDP_AR(Y, copy.deepcopy(starting_params),
                                                 priors=[1*np.eye(D), D],
                                                 num_iter=kwargs['num_iter'] if 'num_iter' in kwargs else 5000,
                                                 return_assignments=True,
                                                 verbose=True,
                                                 chains=chains)
        print()
        print("********************************************************")
        print("******************DONE WITH AR(1)***********************")
        print("********************************************************")
        print()

    if ('exec_model' not in kwargs) or (('exec_model') in kwargs and 1 in kwargs['exec_model']):

        print('RUNNING AR(2) MODEL')

        starting_params['theta'] = [{'A': np.array([np.eye(D)*np.random.normal(0,1,size=D), np.eye(D)*np.random.normal(0,1,size=D)]), 'sigma': 1*np.eye(D)} for i in range(L)]
        starting_params['priors'] = {
            # need to adjust the priors slightly
            'M': np.zeros(shape=(D,D*2), dtype=np.float64),
            'K': 10*np.eye(D*2, dtype=np.float64)
        }

        _, res, assignments_ar2, hamming_ar2 = hdp.sticky_Multi_HDP_AR2(Y, copy.deepcopy(starting_params),
                                                 priors=[np.eye(D), D],
                                                 num_iter=kwargs['num_iter'] if 'num_iter' in kwargs else 5000,
                                                 return_assignments=True,
                                                 verbose=True,
                                                 chains=chains)

        print()
        print("********************************************************")
        print("******************DONE WITH AR(2)***********************")
        print("********************************************************")
        print()

    if ('exec_model' not in kwargs) or (('exec_model') in kwargs and 2 in kwargs['exec_model']):
        print('RUNNING SLDS MODEL')
        if data_choice == 1:
            D = 2
            starting_params['R']     = kwargs['R'] if 'R' in kwargs else 1e-1*np.eye(1)
            starting_params['R0']    = kwargs['R0'] if 'R0' in kwargs else 1e-1*np.eye(1)
            # starting_params['r0']    = kwargs['r0'] if 'r0' in kwargs else 1e1
            starting_params['C']     = np.array([[1, 0]], dtype=np.float64)
            starting_params['D']     = D
            starting_params['theta'] = [{'A': np.eye(D, dtype=np.float64)*np.random.normal(0,1,size=D), 'sigma': 0.1*np.eye(D, dtype=np.float64)} for i in range(L)]
            starting_params['priors'] = {
                'M': np.zeros(shape=(D,D), dtype=np.float64),
                'K': 10*np.eye(D, dtype=np.float64),
            }
            _, z, assignments_slds, hamming_slds = hdp.SLDS_blocked_sampler(Y, copy.deepcopy(starting_params),
                                                  priors=[1e-1*np.eye(D), D],
                                                  num_iter=kwargs['num_iter'] if 'num_iter' in kwargs else 5000,
                                                  verbose=True,
                                                  return_assignments=True,
                                                  chains=chains,
                                                  Y_tilde = np.random.normal(0,1,size=(starting_params['T'], D)).astype(np.float64))
        else:
            starting_params['theta'] = [{'A': np.array(np.eye(D)), 'sigma': 1*np.eye(D)} for i in range(L)]
            starting_params['priors'] = {
                'M': np.zeros(shape=(D,D)),
                'K': 10*np.eye(D)
            }

            _, z, assignments_slds, hamming_slds = hdp.SLDS_blocked_sampler(Y, copy.deepcopy(starting_params),
                                                  priors=[1e-1*np.eye(D), D],
                                                  num_iter=kwargs['num_iter'] if 'num_iter' in kwargs else 5000,
                                                  verbose=True,
                                                  return_assignments=True,
                                                  chains=chains,
                                                  Y_tilde = np.random.normal(0,1,size=(starting_params['T'], D)).astype(np.float64))
        print()
        print("********************************************************")
        print("******************DONE WITH SLDS************************")
        print("********************************************************")
        print()

    print()
    print("********************************************************")
    print("******************DONE ITER: {}, Data #: {}*************".format(iter_number, data_choice))
    print("********************************************************")
    print()

    return {
        'data_choice': data_choice,
        'iter_number': iter_number,
        'hamming_ar1': hamming_ar1,
        'hamming_ar2': hamming_ar2,
        'hamming_slds': hamming_slds
    }

def run_multi_process(outfile, numprocs=4, epochs=100, **kwargs):

    pool = Pool(numprocs)

    results = []

    for i in range(epochs):
        results.append(pool.apply_async(run_experiment_ar, args=(i, 0,), kwds=kwargs))
        results.append(pool.apply_async(run_experiment_ar, args=(i, 1,), kwds=kwargs))
        results.append(pool.apply_async(run_experiment_ar, args=(i, 2,), kwds=kwargs))

    if not os.path.isfile(outfile):
        with open(outfile, 'w') as out_f:
            out_f.write('data,iteration_number,algorithm\n')

    for result in results:
        with open(outfile, 'a') as out_f:
            try:
                res = result.get()
                out_f.write('{data},{iter_no},"AR1",'.format(data=res['data_choice'], iter_no=res['iter_number']) + ','.join(map(str, res['hamming_ar1'])) + '\n')
                out_f.write('{data},{iter_no},"AR2",'.format(data=res['data_choice'], iter_no=res['iter_number']) + ','.join(map(str, res['hamming_ar2'])) + '\n')
                out_f.write('{data},{iter_no},"SLDS",'.format(data=res['data_choice'], iter_no=res['iter_number']) + ','.join(map(str, res['hamming_slds'])) + '\n')
            except:
                print('whoops... something broke')


if __name__ == '__main__':

    if len(sys.argv) < 6:
        print("Need 5 args:\nArg {1} is outfile name\nArg {2} is number processors\nArg {3} is number experiment repeats\nArg {4} is chain lengths\nArg {5} is algorithm num iter")
        sys.exit(1)

    outfile = sys.argv[1]
    num_procs = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    chain_lengths = int(sys.argv[4])
    alg_num_iter = int(sys.argv[5])

    kwargs = {
        'chain_length': chain_lengths,
        'num_iter': alg_num_iter
    }

    run_multi_process(outfile, numprocs=num_procs, epochs=num_epochs, **kwargs)
