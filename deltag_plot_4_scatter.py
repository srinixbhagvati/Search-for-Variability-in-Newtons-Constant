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
x = np.array(data_y)
sigma_i = np.array(data_2)


# In[5]:


def logposterior(theta, data, sigma_i, x):
    lp = logprior(theta) # get the prior

    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta, 
                              data, sigma_i, x)


# In[6]:


def function(x, temp):
    return temp*(10**(-8))*(x**0)

    


# In[25]:


def loglikelihood(theta, data, sigma_i, x):
    
    temp, logsigma = theta

    md = function(x, temp)
    sigma_int  =  10**logsigma
    sigma = ((sigma_i**2) + (sigma_int**2))**0.5

    # check what data and sigma are
    return -0.5 * np.sum(((md - data) / sigma)**2 + np.log(sigma**2))


# In[26]:


def logprior(theta):
    temp, logsigma = theta
    if -2 < temp < 2 and -9 <logsigma <1:
        return 0.0
    return -np.inf


# In[29]:



Nens = 100   # number of ensemble points

logsigmamin = -9
logsigmamax = 1

logsigmaini = np.random.uniform(logsigmamin, logsigmamax, Nens)

tempmin = -2
tempmax = 2

tempini = np.random.uniform(tempmin, tempmax, Nens) # initial c points



inisamples = np.array([tempini, logsigmaini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions


# In[30]:


Nburnin = 500   # number of burn-in samples
Nsamples = 4500  # number of final posterior samples


# In[31]:


import emcee # import the emcee package

print('emcee version: {}'.format(emcee.__version__))

# for bookkeeping set number of likelihood calls to zero
loglikelihood.ncalls = 0

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
argslist = (data, sigma_i, x)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)


# In[32]:


# pass the initial samples and total number of samples required
t0 = time() # start time
sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress = True);
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain()


# In[33]:


labels = ["$C$ ($10^{-8}$)", "$log_{10}(\sigma_{scatter})$"]

flat_samples = sampler.get_chain(discard = Nburnin, thin = 15, flat=True)
print(flat_samples.shape)


# In[34]:


import corner
fig = corner.corner(flat_samples, labels=labels ,levels=(0.68,0.95,0.99),color = 'b',alpha=0.1,fill_contours = 1,max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16});

fig.savefig("contour4_scatter.pdf")


# In[36]:


temp_best = 0
logsigma_best = 0
for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print (mcmc[1], q[0], q[1], labels[i])
    if i == 0:
        a_best = mcmc[1]
    if i == 1:
        logsigma_best = mcmc[1]

sigma_int_best = 10**logsigma_best
sigma = ((sigma_i**2) + (sigma_int_best**2))**0.5


# In[37]:


def chi2_fn(temp_best, sigma_int_best, x, y, sigma_i):
    sigma = ((sigma_i**2) + (sigma_int_best**2))**0.5
    return (np.sum(((y - function(x,temp_best))/sigma)**2))


# In[38]:


print(chi2_fn(temp_best, sigma_int_best, x, data, sigma_i))
chi2_val = chi2_fn(temp_best, sigma_int_best, x, data, sigma_i)
print(len(data) - 2)


# In[ ]:





# In[39]:


print("temp_best: ", temp_best)
print("sigma_int_best: ", sigma_int_best)
chi2_val_best = chi2_fn(temp_best, sigma_int_best,  x, data, sigma)


# In[40]:


def AIC(chi2, k):
    return chi2 + 2*k

def BIC(chi2, k, n):
    return chi2 + np.log(n)*k


# In[41]:


AIC_val = AIC(chi2_val, 2)
print(AIC_val)

BIC_val = BIC(chi2_val, 2, len(data))
print(BIC_val)


# In[42]:



temp = np.average(data, weights = 1/(sigma**2))
print(temp)

AIC_const = AIC(loglike_const(), 1)
print(AIC_const)

BIC_const = BIC(loglike_const(), 1, len(data))
print(BIC_const)


# In[43]:


sigma = ((sigma_i**2) + ((sigma_int_best**2)*10**(-18)))**0.5
print(np.sum(((data - const_fn(x, temp))/sigma)**2))

print(len(data) - 2)


# In[116]:


print("The delta AIC values are: ", AIC_val - AIC_const)
print("The delta BIC values are: ", BIC_val - BIC_const)


# In[117]:


-len(data)*np.log(chi2_val_best/len(data)) + 2*2 + (2*2*3/(len(data)-2-1))


# In[ ]:





# In[ ]:





# In[ ]:




