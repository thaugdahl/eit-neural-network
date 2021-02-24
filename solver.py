# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:10:59 2021

@author: Trygve
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
from numpy import genfromtxt
import equations as eq


def rk4step(tn,xn,f,h):
    k1 = f(tn,xn)
    k2 = f(tn+0.5*h,xn+0.5*h*k1)
    k3 = f(tn+0.5*h,xn+0.5*h*k2)
    k4 = f(tn+h,xn+h*k3)
    xnp1 = xn + h*(k1+2*k2+2*k3+k4)/6
    return xnp1

def plotSolution(dataset,firstRow,lastRow,style="standard"):
    x = dataset[firstRow:lastRow,0:2]
    if style.lower() == "phaseportrait" or style == "pp":
        plt.plot(x[:,0],x[:,1])
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        plt.plot(x[:,0])
        plt.plot(x[:,1])
        plt.xlabel("t")
        plt.ylabel("Solution")
    plt.show()
    
def rk4(f,t,T,h,x0,y0,run):
    N = int(np.ceil(T/h))
    n = 0 
    x = np.zeros((N+1,4),dtype=float)
    x[0,0] = x0[run]
    x[0,1] = y0[run]
    x[:,3] = run
    
    while n<N:
        x[n+1,:-2] = rk4step(t,x[n,:-2],f,h)
        x[n+1,2] = t
        t += h
        n += 1
    return x

def generateTrainingData(saveResults=False):
    # Saving results
    if saveResults:
        fileName = input("Enter a file name to save the training data to be generated: ")
        choice = ""
        while (path.exists(fileName)) and not(choice.lower() == "o"):
            choice = input("WARNING: A file with the selected file name already exists. Overwrite existing file [o] or create new [n]?")
            if choice.lower() == "n":
                fileName = input("Enter a file name to save the training data to be generated: ")
        print("Data will be saved in file ",fileName,"\n")
    # Arrays containing values for Lotka Volterra (LV) parameters
    #alpha = np.array([[0.1],[0.2],[0.3]])
    #beta = np.array([[0.05],[0.1],[0.15]])
    #gamma = np.array([[1.1],[1.2],[1.25]])
    #delta = np.array([[0.1],[0.2],[0.4]])
    # NOTE: alpha, beta, delta, gamma must be inserted in LVparams in this order!
    #LVparams = np.hstack((alpha,beta,delta,gamma))
    # Infer number of runs
    #LVparams_dims = np.shape(LVparams)
    
    # Set other parameters
    f = eq.duffing
    t = 0.0
    T = 200
    h = 0.05
    x0 = np.array([2.0,5.0,1.0])
    y0 = np.array([20.0,15.0,10.0])
    nRuns = x0.size
    dataset = np.zeros((0,4),dtype=float)
    for run in range(int(nRuns)):
        x = rk4(f,t,T,h,x0,y0,run)
        dataset = np.vstack((dataset,x))
    if saveResults:
        np.savetxt(fileName,dataset,delimiter="\t")
        print("Data has been saved in file ",fileName,"\n")
    return dataset
# Generate dataset and set saveResults to true or false
dataset = generateTrainingData(saveResults=False)
# Plot (segment of) solution
firstRow = 0
lastRow = 1000
plotSolution(dataset,firstRow,lastRow,'pp')

def tsv2arr(filename):
    return genfromtxt(filename, delimiter='\t')
x = tsv2arr("eit_test.tsv")