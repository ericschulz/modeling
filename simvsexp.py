from __future__ import print_function, division
import numpy as np
import pandas as pd
import GPy
import json
from scipy import signal
from utilities import *
import os
import sys
import pickle 
import random
import bayesian_optimization
import acquisition_functions
from funcs import *

# Set paths

rpath = '/home/ucabchu/Scratch/modeling/exp1_modeling/rasters/'
wpath = '/home/ucabchu/Scratch/modeling/exp1_modeling/'

setups = [f[5:-5] for f in os.listdir(rpath) if f[0] != '.' and f[0] != 'n' and f[0:5] != 'intro' and f[-4:] == 'json']

l = setups.size

# set parameter values and which_model (can convert to taking command line arguments)

# which_model
dyna = False

taskid =int(os.environ['SGE_TASK_ID'])

f = f = np.linspace(1.6, 4, 25)
part = np.linspace(0, 100)
n = np.linspace(0.05, 0.14, 10)
e = np.linspace(0.003, 0.01, 8)
p = np.linspace(0.6, 0.8, 5)
aG, bG, cG, dG, eG = np.meshgrid(f, n, e, p, part)
aG = aG.flatten() 
bG = bG.flatten()
cG = cG.flatten()
dG = dG.flatten()
eG = eG.flatten()
all=pd.DataFrame({'f':aG, 'n':bG, 'e':cG, 'p':dG, 'part':eG})

factor_low = all.f[taskid-1]
factor_high = 2*factor_low

# noise of the simulation
noise_err_fac = all.n[taskid-1]

#thresholds
entropy_thresh = all.e[taskid-1]
p_thresh = all.p[taskid-1]

# not to be changed for now, set to zero 
p_diff_thresh = 0

index = all.part[taskid-1]


def runAnalysis(setupfn, factor, index, noise_err_fac, entropy_thresh, p_thresh, dyna, 
                store = True, run = True, RAverbose = True):
    
    # acquire data
    noisy_fun, true_fun = raster(rpath, wpath, setupfn, n_disc, n_approx)
    td = round(100*np.mean(true_fun))
    sd = round(100*np.mean(noisy_fun))    
    if RAverbose:
        print("Percentage of ground truth in :", td)
        print("Percentage of simulations in :", sd)
    
    
    # process data
    sd_smooth = 3
    sim_logit, avg_s = dprocess(n_disc, sd_smooth, noisy_fun)
    true_logit, avg_t = dprocess(n_disc, sd_smooth, true_fun)
    error_logit = sim_logit - true_logit
    
    # setup model
    noise_exp = 0.00001 ** 2
#     noise_err = (0.125*np.std(error_logit)) ** 2
    noise_err = (noise_err_fac*np.std(error_logit)) ** 2
    tm = tradeoff_model(sim_logit, true_logit, sd_smooth, noise_exp, noise_err)

    # Some test data
    X, Y = startXY(tm)

    # Define a GP model with the correct noise
    m = GPy.models.GPHeteroscedasticRegression(X, Y, tm.kernel)
    m['.*het_Gauss.variance'] = tm.get_het_noise(m.X)

    #Run ES
    af = acquisition_functions.EntropySearch(m)
    opt = bayesian_optimization.BayesianOptimizer(m, af)

    if run:
        if dyna: opt = runES(tm, factor, entropy_thresh, p_diff_thresh, p_thresh, avg_t, m, af, opt, dyna = True, verbose = False, pverbose = RAverbose)
        else: opt = runES(tm, factor, entropy_thresh, p_diff_thresh, p_thresh, avg_t, m, af, opt, verbose = False, pverbose = RAverbose)
    
        # store runs
        if store:
            
            if dyna:
                sourcefile = '{}psweep_dyna_trial_data{}.csv'.format(wpath, taskid)
            else:
                sourcefile = '{}psweep_full_trial_data{}.csv'.format(wpath, taskid)

            with open(sourcefile, 'a') as outfile:
                Ns, Ne, fx, _ = getData_ESruns(opt = opt)
                outfile.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    index, setupfn, factor, Ne, Ns, fx, noise_err_fac, entropy_thresh, p_thresh))
                
    return



# Run one participant with given params

for s in setups:
    
    All_vals = runAnalysis(s, factor_low, index, noise_err_fac, entropy_thresh, p_thresh, dyna, 
                           RAverbose = False)
    All_vals = runAnalysis(s, factor_high, index, noise_err_fac, entropy_thresh, p_thresh, dyna, 
                           RAverbose = False)


