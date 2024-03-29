import matplotlib.pyplot as plt
import numpy as np
from math import isnan
from sklearn.metrics import mean_squared_error
from settings import *
import scipy.stats as stats



def plot_prediction_accuracy(test_data, predictions, num_vars, title, labels=None):
    """
    Plots a figure with graphs for the predicted data and the actual data.
    :param test_data: The test-data that we wish to predict, as a list of sliding windows
    :param predictions: The predictions to compare with test-data
    :param t_axis: The axis for time in the predicted dataset
    :param num_vars: The number of variables that are relevant in each data-row
    :param title: The title of the figure
    :param labels: The list of labels for each predicted variable
    """
    plt.figure()
    plt.title(title)
    actual_values = test_data[:,-1,:]
    actual_t_axis = actual_values[:,-1] - actual_values[0,-1]
    prediction_t_axis= predictions[:,-1] - predictions[0,-1]
    ymax = max(actual_values[:,:-1].flatten())*1.1
    ymin = min(actual_values[:,:-1].flatten())*1.1
    plt.ylim(ymin,ymax)
    

    if not labels:
        labels = [i for i in range(len(predictions))]

    # Plot all predictions and compare with actual data
    for i in range(0,num_vars):
        # Plot predicted data
        plt.plot(prediction_t_axis, predictions[:,i], label="Predictions {}".format(labels[i]))
        # Plot actual data

        plt.plot(actual_t_axis, actual_values[:,i], label="Actual data {}".format(labels[i]))

    plt.legend()
    plt.show()


def plot_derivatives(taxis, actual, predictions, title=None, labels=None):
    """
    Plots the actual derivatives vs. the given predictions.
    :param x_axis: The values for the x-axis
    :param actual: Array with the actual (approximated) derivatives for each variable
    :param predictions: Array with derivative predictions for each variable
    :param title: The title of the plot
    :param labels: The labels for each of the derivatives
    """
    if not labels:
        labels = [i for i in range(len(actual[1]))]
    plt.figure()
    for i in range(len(actual[1])):
        plt.plot(taxis, actual[:,i], label="Actual values {}".format(labels[i]))
        plt.plot(taxis, predictions[:,i], label="Predictions {}".format(labels[i]))
    if title:
        plt.title(title)
    plt.xlabel("Derivatives")
    plt.legend()
    plt.show()


def plot_training_summary(history, title=None):
    """
    Plots the training loss and validation loss from the history (tensorflow training history).
    :param history: The history
    :param title: A title for the plot
    """
    plt.figure()
    plt.plot(history.history['loss'], label='MSE training data')
    plt.plot(history.history['val_loss'], label='MSE validation data')
    plt.xlabel("Epochs")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_prediction_summary(actual, predictions, labels=None, header=None):
    """
    For each variable in actual/predictions, prints the MSE of the difference between the actual value for a variable
    and the predicted value for the variable.
    :param actual: The list of actual values for the variables
    :param predictions: The list of predicted values for the variables
    :param labels: The labels to use for each variable when printing
    :param header: The header to use for printing
    """
    if np.isnan(predictions).any():
        print("Exploded to infinity :(")
    else:
        if header:
            print(header)
        if not labels:
            labels = [i for i in range(len(predictions[0]))]
        for i in range(len(predictions[0])):
            accuracy = mean_squared_error(actual[::SPARSE,-1,i], predictions[::PREDICTION_TIME_STEP_MULTIPLIER,[i]])
            print("{}-Accuracy: {}".format(labels[i], accuracy))


def plot_confidence_intervals(runs, true, step, num_variables, percentage, titles, injections):
    taxis = runs[0,0,::step,-1]
    
    for i in range(num_variables):
        for j in range(runs.shape[0]):
            inj_runs = runs[j]
            fig,ax = plt.subplots()
            inj_mus = np.mean(inj_runs[:,::step,i],0)
            inj_stdevs = np.std(inj_runs[:,::step,i],0, ddof=1)
            inj_below, inj_above = stats.t.interval(percentage/100, inj_runs.shape[0] -1, inj_mus, inj_stdevs)
            ax.plot(taxis, inj_mus, label = "Gjennomsnitt")
            ax.fill_between(taxis,inj_below,inj_above, alpha=0.2)
            ax.plot(true[:,-1],true[:,i], label= "Sanne verdier")
            ax.plot(taxis,inj_runs[0,:,i], label = "Prediksjon")
            ax.set_xlabel("t")
            ax.set_ylabel("x(t)")
            ymax=max(true[:,:-1].flatten())*2
            ymin=min(true[:,:-1].flatten())*2
            ax.set_ylim((ymin,ymax))
            ax.legend()
            plt.savefig(injections[j][0]+titles[i]+".pdf")
            plt.show()
        plt.show()
        
        
        
        
       