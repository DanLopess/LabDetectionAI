#!/usr/bin/python

import sys
import lib.utils as utils
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
from scipy import stats
from sklearn.metrics import plot_confusion_matrix, classification_report, multilabel_confusion_matrix

warnings.filterwarnings("ignore")


def print_default_arg_format():
    print(
        "Please run the command as: 'python TestMe.py <inputFile> <optionalOutputFile>'"
    )


def prepare_data_classifier_a(df, target_name):
    X = df.drop([target_name], axis=1)
    X = stats.zscore(X)
    y = df[target_name]
    y = y.apply(lambda x: x > 2)
    return X, y


def prepare_data_classifier_b(df, target_name):
    X = df.drop([target_name], axis=1)
    X = stats.zscore(X)
    y = df[target_name]
    return X, y


def prepare_dataset(data):
    data.dropna(
        inplace=True)  # remove missing values (else the model could break)
    # prepare dates
    data['Date'] = data['Date'].astype(str) + ' ' + data['Time'].astype(str)
    data.drop(columns=['Time'], inplace=True)
    format = "%d/%m/%Y %H:%M:%S"
    data['Date'] = data['Date'].apply(
        lambda date: datetime.datetime.strptime(date, format))
    data['Hour'] = data['Date'].apply(lambda time: time.hour)
    data.drop('Date', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def run_classifier_A(data, output):
    x, y = prepare_data_classifier_a(data, target_name="Persons")
    filename = f"models/classifier_A"
    loaded_model = utils.load_model(filename)
    y_pred = loaded_model.predict(x)
    results = utils.get_model_results(y, y_pred)
    print("===== Results for CLASSIFIER A =====")
    utils.print_results('Classifier A - MLP',
                        results,
                        y,
                        y_pred,
                        x,
                        loaded_model,
                        with_chart=True,
                        should_plot=False)
    plt.savefig(f"{output}/classifier_A_confusion_and_metrics.png")


def run_classifier_B(data, output):
    x, y = prepare_data_classifier_b(data, target_name="Persons")
    filename = f"models/classifier_B"
    loaded_model = utils.load_model(filename)
    y_pred = loaded_model.predict(x)
    print("===== Results for CLASSIFIER B =====")
    multilabel_confusion_matrix(y, y_pred)
    plot_confusion_matrix(loaded_model, x, y)
    # _ = utils.get_multiclass_results(y, y_pred)
    print(classification_report(y, y_pred, digits=4))
    plt.title("Classifier B - MLP Confusion Matrix")
    plt.savefig(f"{output}/classifier_B_confusion_matrix.png")


def run(filename, output="results"):
    try:
        data = pd.read_csv(filename)
        data = prepare_dataset(data)
    except:
        # Some csv readers turn the commas into semicolons
        data = pd.read_csv(filename, sep=";")
        data = prepare_dataset(data)
    run_classifier_A(data, output)
    run_classifier_B(data, output)
    plt.show()
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Invalid number of arguments.")
        print_default_arg_format()
        exit()
    else:
        filename = sys.argv[1]
        if (len(sys.argv) == 3):
            output = sys.argv[2]
            run(filename, output)
        else:
            run(filename)