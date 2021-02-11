# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:33:25 2019
copied from: https://aaronschlegel.me/black-scholes-formula-python.html
@author: shirui
"""

import numpy as np
import scipy.stats as si
import math
import matplotlib.pyplot as plt 
import operator

"""
Deltas and explicit solution for euro_vanilla_call
"""
def euro_vanilla_call(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk free interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

def delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1, 0.0, 1.0)

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    n_d1 = si.norm.pdf(d1, 0.0, 1.0)
    return n_d1 / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    n_d1 = si.norm.pdf(d1, 0.0, 1.0)
    N_d2 = si.norm.cdf(d2, 0.0, 1.0)
    
    return -n_d1 * S * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2


def euro_vanilla_put(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put

def sampling_nonlinear(interval,npoints):
    
    LB = interval[0]
    RB = interval[1]
    init = np.linspace(LB,RB,npoints)
    y = pow((init-LB),3)
    inv = np.linspace(np.min(y),np.max(y),npoints)
    return (pow(inv,1/3.0) + LB)

# [S0, K, T, r, sigma] can be frozen with boolean index array
def generate_space(n=100000, frozen=[False for i in range(5)]):
    
    num_per = int(n**(1./(len(frozen) - sum(frozen))))
    
    if frozen[0]:
        S0 = np.array([1.0 for i in range(num_per)])
    else:
        #S0 = sampling_nonlinear([0.01,2],num_per)
        S0 = np.linspace(0.01,2,num_per)
    K = np.linspace(1,1,num_per)
    #m = np.linspace(0.6,1.4,10)
    
    #T = sampling_nonlinear([0.01,0.05],10)
    T = np.linspace(0.01,1.0,num_per)
    
    if frozen[3]:
        r = np.array([0.03 for i in range(num_per)])
    else:
        r = np.linspace(0.01,0.05,num_per)
        
    if frozen[4]:
        sigma = np.array([0.32 for i in range(num_per)])
    else:
        sigma = np.linspace(0.3,1.0,num_per)
    
    dataset = np.concatenate((S0,K,T,r,sigma),axis=0)
    return np.reshape(dataset,(np.size(S0),-1),order='F')


def get_call_option_dataset(frozen=[False for i in range(5)]):
    
    dataset = generate_space(frozen=[False, True, False, True, True])
    
    nc = [len(set(dataset[:,i])) for i in range(np.shape(dataset)[1])]
    n = np.product(nc)
    
    option_call = np.zeros((n,6))
    i=0
    for S0 in list(set(dataset[:,0])):
        for K in list(set(dataset[:,1])):
            for T in list(set(dataset[:,2])):
                for r in list(set(dataset[:,3])):
                    for sigma in list(set(dataset[:,4])):
                        temp = euro_vanilla_call(S0, K, T, r, sigma)
                        option_call[i,:] = [S0,K,T,r,sigma, temp]
                        i = i+1
    return option_call


# Creates stochastically random dataset based off brownian motion with
# fixed interest rate and volatility
def get_brownian_call_option_dataset(length, k=1, r=.03, sigma=.65, seed=-1):
    if seed != -1:
        np.random.seed(seed)
    
    # Generate time steps
    T = np.linspace(0.01, 1.0, length)
    
    # Generate initial stock value stochastically
    S = np.zeros((length))
    S[0] = np.random.rand() * 2
    val = np.zeros((length))
    val[0] = euro_vanilla_call(S[0], k, T[0], r, sigma)
    
    # Generate via brownian motion
    for i in range(1, length):
        d_t = T[i] - T[i-1]
        z = np.random.normal()
        S[i] = S[i-1] * np.exp((r - .5*sigma**2)*d_t + sigma * np.sqrt(d_t)*z)
        # Find value of option
        val[i] = euro_vanilla_call(S[i], k, T[i], r, sigma)
        
    # Generate formatted dataset
    option_call = np.zeros((length, 6))
    option_call[:, 0] = S
    option_call[:, 1] = k
    option_call[:, 2] = T
    option_call[:, 3] = r
    option_call[:, 4] = sigma
    option_call[:, 5] = val
    
    return option_call
    

def normal_pdf(y, mean=0, sigma=1.0):
	numerator = math.exp((-1 * math.pow((y - mean), 2)) / 2 * math.pow(sigma, 2))
	denominator =  math.sqrt(2.0 * math.pi) * sigma
	return numerator / denominator


def _vega(S, K, T, r, sigma):
	""" Calculate derivative of option price with respect to volatility
		vega = s * tau^(1/2) * N(d1)

	Args:
		S (float): stock price
		K (float): strike price
		r (float): risk-free interest rate
		sigma (float: standard deviation of log returns (volatility)
		tau (float): time to option expiration expressed in years
	"""

	d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

	return S * math.sqrt(T) * normal_pdf(d1)

def implied_volatility(S, K, T, r, mp, option_type="call", precision=1e-5, iterations=1000):
	""" Newton-Raphson method of successive approximations to find implied volatility
    mp : option market price
	"""

	# initial estimation
	sigma = 0.5
	for i in range(0, iterations):
		print (i)
		# price of an option as per Black Scholes
		bs_p = euro_vanilla_call(S, K, T, r, sigma)
		diff = mp - bs_p
		# check if difference is acceptable
		if (operator.abs(diff) < precision):
            
			return sigma

		vega = _vega(S, K, T, r, sigma)
        
		print(vega)
        
		# update sigma with addition of difference divided by derivative 
		sigma = sigma + (diff / vega)
		print(sigma)

	# closest estimation
	return sigma


                    
if __name__ == "__main__":
    
    #option_call_dataset = get_call_option_dataset(frozen=[False, True, False, True, True])
    option_call_dataset = get_brownian_call_option_dataset(100000)
    
    plt.hist(option_call_dataset[:,5])
    np.savetxt('option_call_dataset_brownian.txt',option_call_dataset,delimiter=',')
        
