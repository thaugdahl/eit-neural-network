import matplotlib.pyplot as plt
from math import isnan


def plot_prediction_accuracy(test_data, predictions, time_step, num_vars):
    """
    Plots a figure with graphs for the predicted data and the actual data.
    :param test_data: The test-data that we wish to predict
    :param predictions:
    :param time_step:
    :param num_vars:
    :return:
    """
    plt.figure()
    x_axis = [time_step * i for i in range(len(test_data))]
    valid_predictions = [i for i in predictions if not isnan(i[0]) and not isnan(i[1])]
    actual_values = [i[-1] for i in test_data]
    prediction_axises = []
    for i in range(num_vars):
        prediction_axises.append([j[i] for j in valid_predictions])

    # Limit plot y-values
    largest_val = float("-inf")
    for i in range(len(prediction_axises)):
        if prediction_axises[i][0] > largest_val:
            largest_val = prediction_axises[i][0]
    plt.ylim(largest_val * (-5), largest_val * 5)

    # Plot all predictions and compare with actual data
    for i in range(len(prediction_axises)):
        # Plot predicted data
        plt.plot(x_axis[:len(prediction_axises[i])], prediction_axises[i], label="Predictions {}".format(i))
        # Plot actual data
        actual = [j[i] for j in actual_values]
        plt.plot(x_axis, actual, label="Actual data {}".format(i))

    plt.legend()
    plt.show()

