#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from time import time


# In[2]:


data_full = np.genfromtxt('data_csv.csv', delimiter = ',')
#print(data_full)

BRST_y, BRST_1, BRST_2 = zip(*np.genfromtxt('BRST.csv', delimiter = ','))
print(BRST_1)
DUMO_y, DUMO_1, DUMO_2 = zip(*np.genfromtxt('DUMO.csv', delimiter = ','))
#print(DUMO)
GRAS_y, GRAS_1, GRAS_2 = zip(*np.genfromtxt('GRAS.csv', delimiter = ','))
#print(GRAS)
IAIX_y, IAIX_1, IAIX_2 = zip(*np.genfromtxt('IAIX.csv', delimiter = ','))
#print(IAIX)
LARZ_y, LARZ_1, LARZ_2 = zip(*np.genfromtxt('LARZ.csv', delimiter = ','))
#print(LARZ)
LRCH_y, LRCH_1, LRCH_2 = zip(*np.genfromtxt('LRCH.csv', delimiter = ','))
#print(LRCH)
MARS_y, MARS_1, MARS_2 = zip(*np.genfromtxt('MARS.csv', delimiter = ','))
#print(MARS)
NALO_y, NALO_1, NALO_2 = zip(*np.genfromtxt('NALO.csv', delimiter = ','))
#print(NALO)
NYAL_y, NYAL_1, NYAL_2 = zip(*np.genfromtxt('NYAL.csv', delimiter = ','))
#print(NYAL)
SOUL_y, SOUL_1, SOUL_2 = zip(*np.genfromtxt('SOUL.csv', delimiter = ','))
#print(SOUL)
STJ9_y, STJ9_1, STJ9_2 = zip(*np.genfromtxt('STJ9.csv', delimiter = ','))
#print(STJ9)
STRE_y, STRE_1, STRE_2 = zip(*np.genfromtxt('STRE.csv', delimiter = ','))
#print(STRE)
WLBH_y, WLBH_1, WLBH_2 = zip(*np.genfromtxt('WLBH.csv', delimiter = ','))
#print(WLBH)
GRAS2_y, GRAS2_1, GRAS2_2 = zip(*np.genfromtxt('GRAS2.csv', delimiter = ','))
#print(GRAS2)
WLBH2_y, WLBH2_1, WLBH2_2 = zip(*np.genfromtxt('WLBH2.csv', delimiter = ','))
#print(WLBH2)

data_y = BRST_y + DUMO_y + GRAS_y + IAIX_y + LARZ_y + LRCH_y + MARS_y + NALO_y + NYAL_y + SOUL_y + STJ9_y + STRE_y + WLBH_y + GRAS2_y + WLBH2_y
data_1 = BRST_1 + DUMO_1 + GRAS_1 + IAIX_1 + LARZ_1 + LRCH_1 + MARS_1 + NALO_1 + NYAL_1 + SOUL_1 + STJ9_1 + STRE_1 + WLBH_1 + GRAS2_1 + WLBH2_1
data_2 = BRST_2 + DUMO_2 + GRAS_2 + IAIX_2 + LARZ_2 + LRCH_2 + MARS_2 + NALO_2 + NYAL_2 + SOUL_2 + STJ9_2 + STRE_2 + WLBH_2 + GRAS2_2 + WLBH2_2

data = np.array(data_1)
x_abs = np.array(data_y)
x = x_abs - x_abs.min()
sigma = np.array(data_2)


# In[3]:


fig = plt.figure()


plt.errorbar(BRST_y, BRST_1, BRST_2, fmt = '.', label = 'BRST')
plt.errorbar(DUMO_y, DUMO_1, DUMO_2, fmt = ',', label = 'DUMO')
plt.errorbar(GRAS_y, GRAS_1, GRAS_2, fmt = 'v', label = 'GRAS')
plt.errorbar(IAIX_y, IAIX_1, IAIX_2, fmt = 'o', label = 'IAIX')
plt.errorbar(LARZ_y, LARZ_1, LARZ_2, fmt = '^', label = 'LARZ')
plt.errorbar(LRCH_y, LRCH_1, LRCH_2, fmt = '<', label = 'LRCH')
plt.errorbar(MARS_y, MARS_1, MARS_2, fmt = '>', label = 'MARS')
plt.errorbar(NALO_y, NALO_1, NALO_2, fmt = '1', label = 'NALO')
plt.errorbar(NYAL_y, NYAL_1, NYAL_2, fmt = '2', label = 'NYAL')
plt.errorbar(SOUL_y, SOUL_1, SOUL_2, fmt = '3', label = 'SOUL')
plt.errorbar(STJ9_y, STJ9_1, STJ9_2, fmt = '4', label = 'STJ9')
plt.errorbar(STRE_y, STRE_1, STRE_2, fmt = '8', label = 'STRE')
plt.errorbar(WLBH_y, WLBH_1, WLBH_2, fmt = 's', label = 'WLBH')
plt.errorbar(GRAS2_y, GRAS2_1, GRAS2_2, fmt = 'p', label = 'GRAS2')
plt.errorbar(WLBH2_y, WLBH2_1, WLBH2_2, fmt = 'P', label = 'WLBH2')

plt.legend(bbox_to_anchor=(1.0, 1.025)).get_frame().set_edgecolor('black')

fig  = plt.figure(figsize=(6, 4))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 32
plt.rcParams['axes.linewidth'] = 1.5

plt.rcParams['font.weight'] = 1000
plt.tick_params(labelsize=18)
       

plt.ylabel("$\delta$g/g", fontsize = 20, color = 'black', fontweight=1000)
plt.xlabel("year", fontsize = 20, color = 'black', fontweight=1000)


plt.errorbar(x, data, sigma, fmt = '.')
# In[4]:


def logposterior(theta, data, sigma, x):
    lp = logprior(theta) # get the prior

    
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta, data, sigma, x)


# In[5]:


def function(x, a, Ac, As, w):
    return a*10**(-8) + (Ac*10**(-8))*np.cos(w*x) + (As*10**(-8))*np.sin(w*x)

    


# In[6]:


def loglikelihood(theta, data, sigma, x):
    
    a, Ac, As, w = theta

    md = function(x, a, Ac, As, w)

    # check what data and sigma are
    return -0.5 * np.sum(((md - data) / sigma)**2 + np.log(sigma**2))


# In[7]:


def logprior(theta):
    a, Ac, As, w = theta
    T1 = 0.13689628737
    T2 = 11.4
    if -2 < a < 2 and 0 < Ac < 2 and 0 < As < 2 and 2*np.pi/T2 < w < 2*np.pi/T1:
        return 0.0
    return -np.inf


# In[19]:


Nens = 100   # number of ensemble points

T1 = 0.13689628737
T2 = 11.4
wmin = 2*np.pi/T2
wmax = 2*np.pi/T1

wini = np.random.uniform(wmin, wmax, Nens) # initial c points

amin = -2 # lower range of prior
amax = 2  # upper range of prior

aini = np.random.uniform(amin, amax, Nens) # initial c points

Acmin = 0 # lower range of prior
Acmax = 2  # upper range of prior

Acini = np.random.uniform(Acmin, Acmax, Nens) # initial c points

Asmin = 0 # lower range of prior
Asmax = 2  # upper range of prior

Asini = np.random.uniform(Asmin, Asmax, Nens) # initial c points

inisamples = np.array([aini, Acini, Asini, wini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions


# In[20]:


Nburnin = 500   # number of burn-in samples
Nsamples = 4500  # number of final posterior samples


# In[21]:


import emcee # import the emcee package

print('emcee version: {}'.format(emcee.__version__))

# for bookkeeping set number of likelihood calls to zero
loglikelihood.ncalls = 0

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
argslist = (data, sigma, x)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)


# In[22]:


# pass the initial samples and total number of samples required
t0 = time() # start time
sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress = True);
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain()


# In[23]:


labels = ["a ($10^{-8}$)", "Ac ($10^{-8}$)", "As ($10^{-8}$)", "$\omega$ $(rad/year)$"]

flat_samples = sampler.get_chain(discard = Nburnin, thin = 15, flat=True)
print(flat_samples.shape)


# In[24]:


import corner
fig = corner.corner(flat_samples, labels=labels ,levels=(0.68,0.95,0.99),color = 'b',alpha=0.1,fill_contours = 1,max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16});

#fig = corner.corner(
#    flat_samples, labels=labels);
fig.savefig("corner.pdf")


# In[25]:


a_best = 0
Ac_best = 0
As_best = 0
w_best = 0
for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print (mcmc[1], q[0], q[1], labels[i])
    if i == 0:
        a_best = mcmc[1]
    if i == 1:
        Ac_best = mcmc[1]
    if i == 2:
        As_best = mcmc[1]
    if i==3:
        w_best = mcmc[1]


# In[ ]:





# In[26]:


def chi2_fn(a_best, Ac_best, As_best, w_best, x, y, sigma):
    return np.sum(((y - function(x, a_best, Ac_best, As_best, w_best))/sigma)**2)


# In[27]:


print(chi2_fn(a_best, Ac_best, As_best, w_best, x, data, sigma))
print(len(data) - 4)


# In[28]:


print("a_best: ", a_best)
print("Ac_best: ", Ac_best)
print("As_best: ", As_best)
print("w_best: ", w_best)


# In[ ]:





# In[ ]:





# In[ ]:




