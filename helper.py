import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def print_green_bold(msg):
    """
    Print a message in green and bold
    :param msg: a string containing the message to print
    :return: None
    """
    print("\033[1m\033[92m" + msg + "\033[0m")
def print_yellow_bold(msg):
    """
    Print a message in yellow and bold
    :param msg: a string containing the message to print
    :return: None
    """
    print("\033[1m\033[93m" + msg + "\033[0m")
def calculate_confidence_interval(data):
    """
    Calculate the confidence interval for a given data set
    :param data: a list of numbers
    :return: a tuple of the lower and upper bounds of the confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    std_error = std_dev / np.sqrt(n)
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error
    return ci_lower, ci_upper

def graph_time_series(df):
    """
    Graph time series data
    :param df: a pandas dataframe
    :return: None
    """
    date = df["Date"]
    for col in df.columns[1:]:
        plt.plot(date, df[col], label=col)
    plt.legend()
    plt.show()

def test_rf_model(x_test, y_test, model):
    """
    Test the Random Forest model
    :param x_test: test features
    :param y_test: test target
    :param model: trained Random Forest model
    :return: None
    """
    y_pred = model.predict(x_test)
    df_actual_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    return None


def test_et_model(x_test, y_test, model):
    """
    Test the Extra Trees Regressor model
    :param x_test: test features
    :param y_test: test target
    :param model: trained Extra Trees Regressor model
    :return: None
    """
    y_pred = model.predict(x_test)
    df_actual_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    return None

def plot_correlation_matrix(df):
    """
    Plot correlation matrix with values in the box
    :param df: a pandas dataframe
    :return: None
    """
    plt.rcParams["figure.figsize"] = [10, 10]
    corr = df.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j, i, round(corr.iloc[i, j], 2), ha="center", va="center", color="w"
            )
    plt.show()