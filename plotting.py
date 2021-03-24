import matplotlib.pyplot as plt
from math import isnan
from sklearn.metrics import mean_squared_error
from settings import *


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


def plot_derivatives(x_axis, derivatives, predictions, title=None, labels=None):
    """
    Plots the actual derivatives vs. the given predictions.
    :param x_axis: The values for the x-axis
    :param derivatives: Nested list with derivatives for each variable
    :param predictions: Nested list with derivative predictions for each variable
    :param title: The title of the plot
    :param labels: The labels for each of the derivatives
    """
    if not labels:
        labels = [i for i in range(len(derivatives))]
    plt.figure()
    for i in range(len(derivatives)):
        plt.plot(x_axis, derivatives[i], label="Actual values {}".format(labels[i]))
        plt.plot(x_axis, predictions[i], label="Predictions {}".format(labels[i]))
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
    if header:
        print(header)
    if not labels:
        labels = [i for i in range(len(predictions[0]))]
    for i in range(len(predictions[0])):
        accuracy = mean_squared_error(actual[:,-1,i], predictions[::PREDICTION_TIME_STEP_MULTIPLIER,[i]])
        print("{}-Accuracy: {}".format(labels[i], accuracy))

