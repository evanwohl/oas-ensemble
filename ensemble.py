from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from helper import *
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message="X has feature names, but SelectFromModel was fitted without feature names")
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def load_data(csv_path):
    """
    Load data from csv file
    :param csv_path: a string, path to the csv file
    :return: a pandas dataframe with the loaded data
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    original_columns = df.columns.tolist()
    lags = [x for x in range(50)]

    for lag in lags:
        for col in original_columns[1:]:
            df[f"{col}_t-{lag}"] = df[col].shift(lag)

    for col in original_columns[1:]:
        df[f"{col}_volatility"] = df[col].rolling(window=40).std()

    df["Change"] = (
        df["Non Investment Grade Short Term"].shift(-70)
        - df["Non Investment Grade Short Term"]
    )
    df["Change_Binary"] = (df["Change"] < -0.35).astype(int)
    df = df.iloc[:-70]
    df = df.dropna()
    return df


def create_rf_model():
    """
    Create a Random Forest model
    :return: a Random Forest model
    """
    model = RandomForestClassifier(n_estimators=500, verbose=0, n_jobs=-1)
    return model


def create_et_model():
    """
    Create an Extra Trees Classifier model
    :return: an Extra Trees Classifier model
    """
    model = ExtraTreesClassifier(n_estimators=1200, verbose=0, n_jobs=-1)
    return model


def select_features(x, y, feature_cols):
    """
    Select features using Extra Trees Classifier and SelectFromModel
    :param x: a numpy array of features data
    :param y: a numpy array of target values
    :param feature_cols: a list of feature column names
    :return: a tuple of the selected features and the indices of the selected features
    """
    model = ExtraTreesClassifier(n_estimators=1200, n_jobs=-1)
    x_df = pd.DataFrame(x, columns=feature_cols)
    model.fit(x_df, y)
    selector = SelectFromModel(model, prefit=True, threshold="mean")
    x_new = selector.transform(x_df)
    selected_features = selector.get_support(indices=True)
    return x_new, selected_features


def time_series_analysis(df):
    """
    Graph time series data
    :param df: a pandas dataframe containing the data to plot
    :return: None
    """
    df["True Positive"] = (df["Actual"] == 1) & (df["Ensemble Predicted"] == 1)
    df["False Positive"] = (df["Actual"] == 0) & (df["Ensemble Predicted"] == 1)
    df["Percentage Change"] = -(df["Percentage Change"] / 10)
    plt.plot(df["Date"], df["Confidence"], label="Confidence")
    plt.plot(df["Date"], df["Percentage Change"], label="Percentage Change (-1000bps)")
    plt.scatter(
        df["Date"][df["True Positive"]],
        df["Percentage Change"][df["True Positive"]],
        color="green",
        label="True Positive",
    )
    plt.scatter(
        df["Date"][df["False Positive"]],
        df["Percentage Change"][df["False Positive"]],
        color="red",
        label="False Positive",
    )
    plt.xlabel("Date (Integer Based Indexing)")
    plt.ylabel("Confidence/Percentage Change")
    plt.legend()
    plt.show()


def train_test_split_using_sklearn(df):
    """
    Split the data into training and testing sets using sklearn train_test_split function
    :param df: a pandas dataframe containing the data to split
    :return: a tuple of the training features, testing features, training target, and testing target
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].map(pd.Timestamp.timestamp)
    feature_cols = [col for col in df.columns if "Change" not in col]
    y_col = "Change_Binary"
    x = df[feature_cols]
    y = df[y_col]
    x_selected, selected_features = select_features(x, y, feature_cols)
    x_train, x_test, y_train, y_test = train_test_split(
        x_selected, y, test_size=0.25, shuffle=False
    )
    return x_train, x_test, y_train, y_test, selected_features


def train_rf_model(x_train, y_train):
    """
    Train the Random Forest model
    :param x_train: training features represented as a numpy array
    :param y_train: training target represented as a numpy array
    :return: trained Random Forest model
    """
    model = create_rf_model()
    model.fit(x_train, y_train)
    return model


def train_et_model(x_train, y_train):
    """
    Train the Extra Trees Classifier model
    :param x_train: training features represented as a numpy array
    :param y_train: training target represented as a numpy array
    :return: trained Extra Trees Classifier model
    """
    model = create_et_model()
    model.fit(x_train, y_train)
    return model


def train_ensemble_model(x_train, y_train, rf_predictions, et_predictions):
    """
    Train the ensemble model
    :param x_train: training features represented as a numpy array
    :param y_train: training target represented as a numpy array
    :param rf_predictions: predictions from Random Forest model
    :param et_predictions: predictions from Extra Trees Classifier model
    :return: trained model
    """
    ensemble_features = np.column_stack((rf_predictions, et_predictions))
    ensemble_model = LogisticRegression()
    ensemble_model.fit(ensemble_features, y_train)
    return ensemble_model


def plot_confidence_vs_change(df):
    """
    Plot confidence vs percentage change
    :param df: a pandas dataframe containing the data to plot
    :return: None
    """
    df = df[df["Confidence"] > 0.5]
    df["True Positive"] = (df["Actual"] == 1) & (df["Ensemble Predicted"] == 1)
    df["False Positive"] = (df["Actual"] == 0) & (df["Ensemble Predicted"] == 1)
    plt.scatter(
        df["Confidence"][df["True Positive"]],
        df["Percentage Change"][df["True Positive"]],
        color="green",
        label="True Positive",
    )
    plt.scatter(
        df["Confidence"][df["False Positive"]],
        df["Percentage Change"][df["False Positive"]],
        color="red",
        label="False Positive",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Percentage Change")
    plt.title("Confidence vs Percentage Change")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_ensemble_model(
    x_test, y_test, ensemble_model, rf_model, et_model, confidence_threshold, df
):
    """
    Test the ensemble model
    :param x_test: test features data set represented as a numpy array
    :param y_test: test target data set represented as a numpy array
    :param ensemble_model: trained ensemble model
    :param rf_model: trained Random Forest model
    :param et_model: trained Extra Trees Classifier model
    :param confidence_threshold: the confidence threshold for filtering represented as a float
    :return: accuracy, AUC, and confidence scores for the test set represented as a tuple
    """
    rf_predictions = rf_model.predict_proba(x_test)[:, 1]
    et_predictions = et_model.predict_proba(x_test)[:, 1]
    ensemble_features = np.column_stack((rf_predictions, et_predictions))
    ensemble_predictions = ensemble_model.predict(ensemble_features)
    ensemble_confidences = ensemble_model.predict_proba(ensemble_features)[:, 1]

    df_actual_vs_pred = pd.DataFrame(
        {
            "Actual": y_test,
            "Ensemble Predicted": ensemble_predictions,
            "Confidence": ensemble_confidences,
            "Percentage Change": df["Change"].iloc[-len(y_test) :],
            "Date": df["Date"].iloc[-len(y_test) :],
        }
    )
    print(df_actual_vs_pred.to_string())
    plot_confidence_vs_change(df_actual_vs_pred)
    time_series_analysis(df_actual_vs_pred)

    accuracy = accuracy_score(y_test, ensemble_predictions)
    auc = roc_auc_score(y_test, ensemble_predictions)
    filtered_df = df_actual_vs_pred[
        df_actual_vs_pred["Confidence"] > confidence_threshold
    ]
    if len(filtered_df) > 0:
        filtered_accuracy = accuracy_score(
            filtered_df["Actual"], filtered_df["Ensemble Predicted"]
        )
    else:
        filtered_accuracy = None

    return accuracy, auc, ensemble_confidences, filtered_accuracy

def main(df, confidence_threshold):
    """
    Main function to train and test the ensemble model
    :param df: a pandas dataframe containing the data for training and testing
    :param confidence_threshold: the confidence threshold for filtering, float
    :return: a tuple of the accuracies, AUCs, selected features, confidence scores, and filtered accuracies
    """
    (
        x_train,
        x_test,
        y_train,
        y_test,
        selected_features,
    ) = train_test_split_using_sklearn(df)
    print_green_bold("Data Loaded")
    print_green_bold(
        "--------------------------------------------------------------------------------------------------------------------------------"
    )
    print_green_bold(
        f"{len(selected_features)} features selected out of {len(df.columns) - 1} features"
    )
    accuracies = []
    aucs = []
    confidence_scores = []
    filtered_accuracies = []
    print_yellow_bold(f"Training Started")
    rf_model = train_rf_model(x_train, y_train)
    et_model = train_et_model(x_train, y_train)

    rf_predictions_train = rf_model.predict_proba(x_train)[:, 1]
    et_predictions_train = et_model.predict_proba(x_train)[:, 1]

    ensemble_model = train_ensemble_model(
        x_train, y_train, rf_predictions_train, et_predictions_train
    )

    accuracy, auc, confidences, filtered_accuracy = test_ensemble_model(
        x_test, y_test, ensemble_model, rf_model, et_model, confidence_threshold, df
    )
    accuracies.append(accuracy)
    aucs.append(auc)
    confidence_scores.extend(confidences)
    if filtered_accuracy is not None:
        filtered_accuracies.append(filtered_accuracy)
    print_green_bold(f"Training Completed")
    return accuracies, aucs, selected_features, confidence_scores, filtered_accuracies


def graph_confidence_score_vs_change(df):
    """
    Graph confidence score vs change
    :param df: a pandas dataframe
    :return: None
    """
    plt.scatter(df["Confidence"], df["Actual"])
    plt.xlabel("Confidence")
    plt.ylabel("Change")
    plt.show()


if __name__ == "__main__":
    df = load_data("fred_data.csv")
    confidence_threshold = 0.5
    (
        accuracies,
        aucs,
        selected_features,
        confidence_scores,
        filtered_accuracies,
    ) = main(df, confidence_threshold)
    print_green_bold(
        "--------------------------------------------------------------------------------------------------------------------------------"
    )
    print_green_bold(f"Accuracy (Entire Test Set): {np.mean(accuracies)}")
    print_green_bold(f"AUC (Entire Test Set): {np.mean(aucs)}")
    if filtered_accuracies:
        print_green_bold(
            f"Filtered Accuracy (Confidence > {confidence_threshold}): {np.mean(filtered_accuracies)}"
        )
    else:
        print_green_bold(
            f"No predictions with confidence greater than {confidence_threshold}."
        )
