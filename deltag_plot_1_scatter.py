#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from time import time


# In[3]:


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
sigma_i = np.array(data_2)
#print(x[90], x_abs[90], x_abs.min())


# In[49]:


def logposterior(theta, data, sigma_i, x):
    lp = logprior(theta) # get the prior

    if not np.isfinite(lp):
        return -np.inf

    return lp + loglikelihood(theta, data, sigma_i, x)


# In[50]:


def function(x, a, Ac, As, w):
    return a*10**(-8) + (Ac*10**(-8))*np.cos(w*x) + (As*10**(-8))*np.sin(w*x)

    


# In[51]:


def loglikelihood(theta, data, sigma_i, x):
    
    a, Ac, As, w, logsigma = theta

    md = function(x, a, Ac, As, w)
    sigma_int = 10**(logsigma)
    sigma = ((sigma_i**2) + (sigma_int**2))**0.5
    # check what data and sigma are
    return -0.5 * np.sum(((md - data) / sigma)**2 + np.log(sigma**2))


# In[67]:


def logprior(theta):
    a, Ac, As, w, logsigma = theta
    T1 = 0.13689628737
    T2 = 11.4
    if -2 < a < 2 and -2 < Ac < 2 and -2 < As < 2 and 2*np.pi/T2 < w < 2*np.pi/T1 and -9 < logsigma < 1:
        return 0.0
    return -np.inf


# In[68]:


Nens = 100   # number of ensemble points

logsigmamin = -9
logsigmamax = 1

logsigmaini = np.random.uniform(logsigmamin, logsigmamax, Nens)
#print(sigma_intini)

T1 = 0.13689628737
T2 = 11.4
wmin = 2*np.pi/T2
wmax = 2*np.pi/T1

wini = np.random.uniform(wmin, wmax, Nens) # initial c points

amin = -2 # lower range of prior
amax = 2  # upper range of prior

aini = np.random.uniform(amin, amax, Nens) # initial c points

Acmin = -2 # lower range of prior
Acmax = 2  # upper range of prior

Acini = np.random.uniform(Acmin, Acmax, Nens) # initial c points

Asmin = -2 # lower range of prior
Asmax = 2  # upper range of prior

Asini = np.random.uniform(Asmin, Asmax, Nens) # initial c points

inisamples = np.array([aini, Acini, Asini, wini, logsigmaini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions


# In[69]:


Nburnin = 500   # number of burn-in samples
Nsamples = 4500  # number of final posterior samples


# In[70]:


import emcee # import the emcee package

print('emcee version: {}'.format(emcee.__version__))

# for bookkeeping set number of likelihood calls to zero
loglikelihood.ncalls = 0

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
argslist = (data, sigma_i, x)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)


# In[71]:


# pass the initial samples and total number of samples required
t0 = time() # start time
sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress = True);
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain()


# In[72]:


labels = ["$a$ ($10^{-8}$)", "$A_c$ ($10^{-8}$)", "$A_s$ ($10^{-8}$)", "$\omega$ $(rad/year)$", "$log_{10}(\sigma_{scatter})$"]

flat_samples = sampler.get_chain(discard = Nburnin, thin = 15, flat=True)
print(flat_samples.shape)


# In[73]:


import corner
fig = corner.corner(flat_samples, labels=labels ,levels=(0.68,0.95,0.99),color = 'b',alpha=0.1,fill_contours = 1,max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16});

#fig = corner.corner(
#    flat_samples, labels=labels);
fig.savefig("contour1_scatter.pdf")


# In[74]:


a_best = 0
Ac_best = 0
As_best = 0
w_best = 0
logsigma_best = 0
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
    if i==4:
        logsigma_best = mcmc[1]
        
sigma_int_best = 10**(logsigma_best)
        
sigma = ((sigma_i**2) + (sigma_int_best**2))**0.5


# In[75]:


def chi2_fn(a_best, Ac_best, As_best, w_best, sigma_int_best, x, y, sigma_i):
    sigma = ((sigma_i**2) + (sigma_int_best**2))**0.5
    return (np.sum(((y - function(x, a_best, Ac_best, As_best, w_best))/sigma)**2))


# In[76]:


print(chi2_fn(a_best, Ac_best, As_best, w_best, sigma_int_best, x, data, sigma_i))
chi2_val = chi2_fn(a_best, Ac_best, As_best, w_best, sigma_int_best, x, data, sigma_i)
print(len(data) - 5)


# In[77]:


print("a_best: ", a_best)
print("Ac_best: ", Ac_best)
print("As_best: ", As_best)
print("w_best: ", w_best)
print("sigma_int_best: ", sigma_int_best)


# In[78]:


def AIC(chi2, k):
    return chi2 + 2*k

def BIC(chi2, k, n):
    return chi2 + np.log(n)*k

def maxll(x0, func):
    return -func(x0)

def loglike():
    yM=function(x, a_best, Ac_best, As_best, w_best)
    sigma = ((sigma_i**2) + ((sigma_int_best**2)*10**(-18)))**0.5
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                         + (data - yM) ** 2 / sigma ** 2)
# In[79]:




AIC_val = AIC(chi2_val, 5)
print(AIC_val)

BIC_val = BIC(chi2_val, 5, len(data))
print(BIC_val)


# In[80]:




AIC_const = 358.36543168544176
BIC_const = 366.1096672643926


# In[81]:


print("The delta AIC values are: ", AIC_val - AIC_const)
print("The delta BIC values are: ", BIC_val - BIC_const)


# # 

# In[ ]:





# In[ ]:




