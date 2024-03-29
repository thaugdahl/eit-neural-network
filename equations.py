# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:38:53 2021

@author: Trygve
"""

import numpy as np
import matplotlib.pyplot as plt

def pertlotka_volterra(tn,xn):
    alpha = 0.1 # Birth rate of prey in absence of predators 
    beta = 0.05 # Rate of predation 
    gamma = 1.1 # Death rate of predators 
    delta = 0.1 # Growth rate of predators
    
    x = alpha*xn[0] - beta*xn[0]*xn[1]
    
    y = delta*xn[0]*xn[1]  - gamma*xn[1]
    
    return np.array([x,y])

def lotka_volterra(tn,xn):
    alpha = 0.1 # Birth rate of prey in absence of predators 
    beta = 0.05 # Rate of predation 
    gamma = 1.1 # Death rate of predators 
    delta = 0.1 # Growth rate of predators
    
    x = alpha*xn[0] - beta*xn[0]*xn[1] 
    
    y = delta*xn[0]*xn[1]-gamma*xn[1]
    
    
    return np.array([x,y])

def duffing(t,x):
    delta = 1 # Linear dampening
    alpha = 0.5 #Linear stiffness
    beta = 1 #Nonlinear dampening
    gamma =  3 #Driving amplitude
    omega = 0.4 #Driving frequency
    
    f0 = x[1]
    f1 = gamma*np.cos(omega*t) - delta*x[1] - alpha*x[0] - beta*(x[0]**3)*(1 + 0.1*(x[0]**2))
    
    return np.array([f0, f1])

def lorentz(t,x):
    a = 10*(x[1]-x[0])
    b = x[0]*(28-x[2]) - x[1]
    c = x[0]*x[1] - (8/3)*x[2]
    
    return np.array([a,b,c])



