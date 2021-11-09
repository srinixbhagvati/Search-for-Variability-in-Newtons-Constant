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


def function(x, a, A, w, phi):
    return a*(10**(-8)) + A*(10**(-8))*np.cos(w*x + phi)


# In[5]:


def logposterior(theta, data, sigma, x):
    lp = logprior(theta) # get the prior

    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta, data, sigma, x)


# In[6]:


def function(x, a, A, w, phi):
    return a*(10**(-8)) + A*(10**(-8))*np.cos(w*x + phi)

    


# In[7]:


def loglikelihood(theta, data, sigma, x):
    
    a, A, w, phi = theta

    md = function(x, a, A, w, phi)

    # check what data and sigma are
    return -0.5 *np.sum(((md - data) / sigma)**2 + np.log(sigma**2))


# In[8]:


def logprior(theta):
    a, A, w, phi = theta
    T1 = 0.13689628737
    T2 = 11.4
    if -2 < a < 2 and 0 < A < 2 and 2*np.pi/T2 < w < 2*np.pi/T1 and 0 < phi < 2*math.pi:
        return 0.0
    return -np.inf


# In[9]:


Nens = 100   # number of ensemble points

T1 = 0.13689628737
T2 = 11.4
wmin = 2*np.pi/T2
wmax = 2*np.pi/T1

wini = np.random.uniform(wmin, wmax, Nens) # initial c points

amin = -2 # lower range of prior
amax = 2  # upper range of prior

aini = np.random.uniform(amin, amax, Nens) # initial c points

Amin = 0 # lower range of prior
Amax = 2  # upper range of prior

Aini = np.random.uniform(Amin, Amax, Nens) # initial c points

phimin = 0 # lower range of prior
phimax = 2*math.pi  # upper range of prior

phiini = np.random.uniform(phimin, phimax, Nens) # initial c points

inisamples = np.array([aini, Aini, wini, phiini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions


# In[10]:


Nburnin = 500   # number of burn-in samples
Nsamples = 4500  # number of final posterior samples


# In[11]:


import emcee # import the emcee package

print('emcee version: {}'.format(emcee.__version__))

# for bookkeeping set number of likelihood calls to zero
loglikelihood.ncalls = 0

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
argslist = (data, sigma, x)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)


# In[12]:


# pass the initial samples and total number of samples required
t0 = time() # start time
sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress = True);
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain()


# In[13]:


labels = ["a ($10^{-8}$)", "A ($10^{-8}$)", "$\omega$ $(rad/year)$", "$\phi$ $(rad)$"]

flat_samples = sampler.get_chain(discard = Nburnin, thin = 15, flat=True)
print(flat_samples.shape)


# In[14]:


import corner
fig = corner.corner(flat_samples, labels=labels ,levels=(0.68,0.95,0.99),color = 'b',alpha=0.1,fill_contours = 1,max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16});

#fig = corner.corner(
#    flat_samples, labels=labels);
fig.savefig("corner.pdf")


# In[15]:


a_best = 0
A_best = 0
w_best = 0
phi_best = 0
for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print (mcmc[1], q[0], q[1], labels[i])
    if i == 0:
        a_best = mcmc[1]
    if i == 1:
        A_best = mcmc[1]
    if i == 2:
        w_best = mcmc[1]
    if i == 3:
        phi_best = mcmc[1]


# In[16]:


def chi2_fn(a_best, A_best, w_best, phi_best, x, y, sigma):
    return (np.sum(((y - function(x, a_best, A_best, w_best, phi_best))/sigma)**2))


# In[17]:


print(chi2_fn(a_best, A_best, w_best, phi_best, x, data, sigma))
print(len(data) - 4)


# In[18]:


print("a_best: ", a_best)
print("A_best: ", A_best)
print("w_best: ", w_best)
print("phi_best: ", phi_best)


# In[ ]:




