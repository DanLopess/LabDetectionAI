import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, title
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             precision_score, confusion_matrix)
import lib.metrics as metrics
import pickle
import seaborn as sns


def save_system(system_name, system):
    pkl_filename = f"systems/{system_name}.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(system, file)


def load_system(system_name):
    pkl_filename = f"systems/{system_name}.pkl"
    with open(pkl_filename, "rb") as file:
        pickle_system = pickle.load(file)

    return pickle_system


def detect_and_remove_outliers_std(df: pd.DataFrame, k=2, ignore_cols=()):
    cols_averages_std = [(name, np.mean(values), np.std(values))
                         for name, values in df.iteritems()
                         if name not in ignore_cols]
    df_dropped_outliers = df.copy()
    for index, row in df.iterrows():
        for col_avg in cols_averages_std:
            if (isinstance(row[col_avg[0]], (float, int))
                    and np.abs(row[col_avg[0]] - col_avg[1]) > col_avg[2] * k):
                df_dropped_outliers.drop(index=index, inplace=True)
                break
    return df_dropped_outliers


def plot_corr_matrx(data):
    fig = figure(figsize=[12, 12])
    corr_mtx = abs(data.corr())
    heatmap(abs(corr_mtx),
            xticklabels=corr_mtx.columns,
            yticklabels=corr_mtx.columns,
            annot=True,
            cmap='Blues')
    title('Correlation analysis')
    show()


def get_extremes(column: pd.Series, name=""):
    min = np.min(column)
    max = np.max(column)
    print(f'{name} - Min: {min} Max: {max}')
    return min, max


def get_fuzzy_results(y_test, y_pred):
    cf_mtrx = confusion_matrix(y_test, y_pred)
    scores = metrics.get_metrics(pd.DataFrame(cf_mtrx))
    avg_scores = np.mean(scores)
    return scores, avg_scores


def plot_cfx_mtrx(y_test, y_pred, show=True):
    cf_mtrx = confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(cf_mtrx, annot=True, cmap='Blues', fmt='d')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    if show:
        plt.show()


def get_results(y_test, y_pred, show=True):
    plot_cfx_mtrx(y_test, y_pred, show)
    acc, rec, prec, f1 = (
        accuracy_score(y_test, y_pred, normalize=True),
        recall_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    )
    return acc, rec, prec, f1


def prepare_features(data, target_name):
    data = data.copy()
    data[target_name] = data[target_name].apply(lambda x: 1 if x > 2 else 0)
    data.drop(columns=['S1Temp', 'S2Temp', 'S3Temp'], inplace=True)
    data['Light'] = data['S1Light'] + data['S2Light'] + data[
        'S3Light']  # total light
    data.drop(columns=['S1Light', 'S2Light', 'S3Light'], inplace=True)
    data.drop(columns=['PIR1', 'PIR2'], inplace=True)
    data['CO2_Diff'] = (data['CO2'] - data['CO2'].shift(10))
    return data