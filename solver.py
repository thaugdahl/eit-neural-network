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
from tqdm import tqdm
from random import shuffle


def rk4step(tn,xn,f,h):
    """
    Calculates the next timestep using the rungekutta method.
    :param tn: The current value of the t variable
    :param xn: The current position vector
    :param f: The function describing the ODE we wish to solve
    :param h: The discrete timestep
    :return: Returns the position vector of the system at time tn+h
    """
    
    k1 = f(tn,xn)
    k2 = f(tn+0.5*h,xn+0.5*h*k1)
    k3 = f(tn+0.5*h,xn+0.5*h*k2)
    k4 = f(tn+h,xn+h*k3)
    xnp1 = xn + h*(k1+2*k2+2*k3+k4)/6
    return xnp1


def plotSolution(dataset, firstRow, lastRow, style="standard"):
    x = dataset[firstRow:lastRow]
    if style.lower() == "phaseportrait" or style == "pp":
        plt.plot(x[:, 0], x[:, 1])
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        for i in range(dataset.shape[1]-2):
            plt.plot(x[:,-2],x[:, i])
        
    plt.show()
    
    
def rk4(f,t0,T,h,x0,d):
    """
    Calculates a trajecory of a dynamical system with given initial data.
    :param f: The function controlling the dynamical system, \dot x = f(t,x)
    :param t: The initial time of the system
    :param T: The terminal time
    :param h: The discrete timestep
    :param x0: The initial conditions, the position of the system at time t0.
    :param d: The number of spatial dimensions of the system
    """
    N = int(np.ceil((T-t0)/h))
    n = 0 
    x = np.zeros((N+1,d+1),dtype=float)
    x[0,:-1] = x0
    
    t=t0
    
    while n<N:
        x[n+1,:-1] = rk4step(t,x[n,:-1],f,h)
        x[n+1,-1] = t
        t += h
        n += 1
    return x


def generateTrainingData(saveResults=False):
    # Saving results
    if saveResults:
        fileName = input(
            "Enter a file name to save the training data to be generated: ")
        choice = ""
        while (path.exists(fileName)) and not(choice.lower() == "o"):
            choice = input(
                "WARNING: A file with the selected file name already exists. Overwrite existing file [o] or create new [n]?")
            if choice.lower() == "n":
                fileName = input(
                    "Enter a file name to save the training data to be generated: ")
        print("Data will be saved in file ", fileName, "\n")
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
    T = 100
    h = 0.05

    x0 = np.array([[1,1],[0.5,1],[0,1],[-0.5,1],[-1,1],[-1,0.5],[-1,0],[-1,-0.5],[-1,-1]])  
    nRuns,d = x0.shape
    dataset = np.zeros((0,d+2),dtype=float)
    for run in range(int(nRuns)):
        x = rk4(f,t,T,h,x0[run],d)
        runcol = np.full((x.shape[0],1), run)
        x = np.hstack((x,runcol))
        dataset = np.vstack((dataset,x))
    if saveResults:
        np.savetxt(fileName, dataset, delimiter="\t")
        print("Data has been saved in file ", fileName, "\n")
    return dataset




def tsv2arr(filename):
    return genfromtxt(filename, delimiter='\t')
# x = tsv2arr("eit_test.tsv")


if __name__ == '__main__':
    dataset = generateTrainingData(saveResults=True)
    firstRow = 0
    lastRow = 1000000
    plotSolution(dataset,firstRow,lastRow)
"""
# Generate dataset and set saveResults to true or false
dataset = generateTrainingData(saveResults=True)
# Plot (segment of) solution
firstRow = 0
lastRow = 10000000
plotSolution(dataset,firstRow,lastRow,'pp')
"""

