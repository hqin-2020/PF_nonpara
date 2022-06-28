import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import pickle
import os

from PF_Aso1_0_non_parallel_wcd import *

workdir = os.path.dirname(os.getcwd())
srcdir = os.getcwd()
datadir = workdir + '/data/'
outputdir = workdir + '/output/'

seed = 9

obs_series = pd.read_csv(datadir + 'data.csv', delimiter=',')
obs_series = np.array(obs_series.iloc[:,1:]).T

T = obs_series.shape[1]
N = 30000
Λ_scale = 1
cd_scale = 1
np.random.seed(seed)
start_time = time.time()
H_0_series = []
θ_0_series = []
X_0_series = []
D_0 = obs_series[:,[0]]

for i in range(N):
    θ_0, X_0, H_0, _ = init(D_0, Λ_scale, cd_scale)
    H_0_series.append(H_0)
    θ_0_series.append(θ_0)
    X_0_series.append(X_0)
run_time = time.time() - start_time
print(run_time)    

H_series = [H_0_series]
θ_series = [θ_0_series]
X_series = [X_0_series]
count_series = [np.ones(N)]

H_TEMP_series = [H_0_series]
θ_TEMP_series = [θ_0_series]
X_TEMP_series = [X_0_series]
w_series = [np.ones(N)/N]

for t in tqdm(range(T-1)):
    
    H_temp_series = []
    θ_temp_series = []
    X_temp_series = []
    w_temp_series = []
    
    D_t_next = obs_series[:,[t+1]]

    H_t = H_series[-1]
    X_t = X_series[-1]
    
    for i in range(N):
        
        θ_t_next = update_θ(H_t[i])
        θ_temp_series.append(θ_t_next)

        X_t_next = update_X(D_t_next, X_t[i], θ_t_next)
        X_temp_series.append(X_t_next)

        H_t_next = update_H(X_t_next, X_t[i], H_t[i])
        H_temp_series.append(H_t_next)

        w_t_next = update_ν(D_t_next, X_t[i], θ_t_next)
        w_temp_series.append(w_t_next)
    
    w = w_temp_series/np.sum(w_temp_series)
    try:
        count_all = sp.stats.multinomial.rvs(N, w)
    except:
        for i in range(w.shape[0]):
            if w[i]>(np.sum(w[:-1]) - 1):
                w[i] = w[i] - (np.sum(w[:-1]) - 1)
                break
        count_all = sp.stats.multinomial.rvs(N, w)
    
    H_TEMP_series.append(H_temp_series)
    θ_TEMP_series.append(θ_temp_series)
    X_TEMP_series.append(X_temp_series)
    count_series.append(count_all)
    
    Ht_particle = []
    θt_particle = []
    Xt_particle = []
    
    for i in range(N):
        if count_all[i] != 0:
            for n in range(count_all[i]):
                Ht_particle.append(H_temp_series[i])
                θt_particle.append(θ_temp_series[i])
                Xt_particle.append(X_temp_series[i])
            
    H_series.append(Ht_particle)
    θ_series.append(θt_particle)
    X_series.append(Xt_particle)
    w_series.append(w)

case = 'actual data wcd, seed = ' + str(seed) + ', T = ' + str(T) + ', N = ' + str(N) + ', Λ_scale = ' + str(Λ_scale) + ', cd_scale = ' + str(cd_scale)
try: 
    casedir = outputdir + case  + '/'
    os.mkdir(casedir)
except:
    casedir = outputdir + case  + '/'
    
for t in tqdm(range(T)):
    with open(casedir + 'H_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(H_series[t], f)
    with open(casedir + 'θ_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(θ_series[t], f)
    with open(casedir + 'X_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(X_series[t], f)
    with open(casedir + 'count_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(count_series[t], f)
    with open(casedir + 'H_TEMP_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(H_TEMP_series[t], f)
    with open(casedir + 'θ_TEMP_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(θ_TEMP_series[t], f)
    with open(casedir + 'X_TEMP_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(X_TEMP_series[t], f)
    with open(casedir + 'w_series_'+str(t)+'.pkl', 'wb') as f:
           pickle.dump(w_series[t], f)