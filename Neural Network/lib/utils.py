from audioop import mul
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             precision_score, plot_confusion_matrix,
                             confusion_matrix)
import pickle
from imblearn.over_sampling import SMOTENC, SMOTE
from operator import itemgetter
import lib.metrics as metrics


def run_with_timer(func):
    st = time.time()
    result = func()
    duration = time.time() - st
    print(f"Function took {duration:.2f} seconds to complete.")
    return result, duration


def save_model(model_name, model):
    pkl_filename = f"{model_name}.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(model, file)


def load_model(model_name):
    pkl_filename = f"{model_name}.pkl"
    with open(pkl_filename, "rb") as file:
        pickle_model = pickle.load(file)

    return pickle_model


def plot_evaluation_results(ax, acc, rec, prec, f1, model_name="Model's"):
    evaluation = {
        "Accuracy": float(acc),
        "Recall": float(rec),
        "Precision": float(prec),
        "F1-score": float(f1),
    }

    names = list(evaluation.keys())
    values = list(evaluation.values())
    ax.bar(range(len(evaluation)), values, tick_label=names)
    ax.set_title(f"{model_name} performance metrics")

    # Make some labels.
    rects = ax.patches
    labels = []
    for value in evaluation.values():
        labels.append(f"{value:.4f}")

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.02,
            label,
            ha="center",
            va="bottom",
        )


def get_model_results(y_test, y_pred):
    acc, rec, prec, f1 = (
        accuracy_score(y_test, y_pred, normalize=True),
        recall_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    )
    return acc, rec, prec, f1


def print_results(model_name,
                  results,
                  y_test,
                  y_pred,
                  x_test,
                  model,
                  with_chart=False, should_plot=True):
    acc, rec, prec, f1 = results
    print(f"{model_name} accuracy : {acc}")
    print(f"{model_name} recall : {rec}")
    print(f"{model_name} precision : {prec}")
    print(f"{model_name} f1 score : {f1}")

    if with_chart:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        plot_evaluation_results(axes[0], acc, rec, prec, f1, model_name)
        plot_confusion_matrix(model, x_test, y_test, ax=axes[1])
        axes[1].set_title(f"{model_name} Confusion Matrix")
        if should_plot:
            plt.show()


def run_class_baseline(x_train,
                       y_train,
                       x_val,
                       y_val,
                       random_state=42,
                       ignore_results=False):
    neigh = KNeighborsClassifier(n_neighbors=3)
    mlp = MLPClassifier(solver="lbfgs",
                        hidden_layer_sizes=(5),
                        random_state=random_state)
    if not ignore_results:
        neigh.fit(x_train, y_train.ravel())
        mlp.fit(x_train, y_train)
        pred_knn = neigh.predict(x_val)
        knn_r = get_model_results(y_val, pred_knn)
        pred_mlp = mlp.predict(x_val)
        mlp_r = get_model_results(y_val, pred_mlp)
        return (neigh, knn_r), (mlp, mlp_r)

    else:
        return neigh, mlp


def split_dataset(X, y, test_size=0.1, validation_size=0.20, random_state=42):
    total_test_ratio = test_size + validation_size
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=total_test_ratio, random_state=random_state)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_size / total_test_ratio,
        random_state=random_state,
    )
    return x_train, y_train, x_valid, y_valid, x_test, y_test


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


def detect_and_fill_outliers_std(df: pd.DataFrame, k=2, ignore_cols=()):
    cols_averages_std = [(name, np.mean(values), np.std(values))
                         for name, values in df.iteritems()
                         if name not in ignore_cols]
    df_detected_outliers = df.copy()
    for index, row in df.iterrows():
        for col_avg in cols_averages_std:
            if (isinstance(row[col_avg[0]], (float, int))
                    and np.abs(row[col_avg[0]] - col_avg[1]) > col_avg[2] * k):
                df_detected_outliers.loc[index, col_avg[0]] = np.nan

    df_detected_outliers.fillna(method="ffill", inplace=True)
    df_detected_outliers.fillna(method="bfill", inplace=True)
    return df_detected_outliers


def get_variable_types(df: pd.DataFrame) -> dict:
    variable_types: dict = {
        "Numeric": [],
        "Binary": [],
        "Date": [],
        "Symbolic": []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types["Binary"].append(c)
            df[c].astype("bool")
        elif df[c].dtype in ["datetime64", "datetime64[ns]"]:
            variable_types["Date"].append(c)
        elif df[c].dtype == "int":
            variable_types["Numeric"].append(c)
        elif df[c].dtype == "float":
            variable_types["Numeric"].append(c)
        else:
            df[c].astype("category")
            variable_types["Symbolic"].append(c)
    return variable_types


def underSample(df_negatives, df_positives, values):
    df_neg_sample = pd.DataFrame(df_negatives.sample(len(df_positives)))
    df_under = pd.concat([df_positives, df_neg_sample], axis=0)
    values["UnderSample"] = [len(df_positives), len(df_neg_sample)]
    return df_under


def overSample(df_negatives, df_positives, values):
    df_pos_sample = pd.DataFrame(
        df_positives.sample(len(df_negatives), replace=True))
    df_over = pd.concat([df_pos_sample, df_negatives], axis=0)
    values["OverSample"] = [len(df_pos_sample), len(df_negatives)]
    return df_over


def smote(df, class_var, values, negative_class, positive_class):
    RANDOM_STATE = 42
    vars = get_variable_types(df)["Symbolic"]
    if len(vars) > 0:
        smote = SMOTENC(
            sampling_strategy="minority",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            categorical_features=[
                df.columns.get_loc(c) for c in vars if c in df
            ],
        )
    else:
        smote = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
    y = df.pop(class_var).values
    X = df.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = pd.concat(
        [pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df.columns) + [class_var]
    if len(vars) > 0:
        df_smote = df_smote.infer_objects()
    smote_target_count = pd.Series(smote_y).value_counts()
    values["SMOTE"] = [
        smote_target_count[positive_class],
        smote_target_count[negative_class],
    ]
    return df_smote


def apply_balancing(X, y, class_var):
    df = X.copy()
    df["Persons"] = y
    target_count = y.value_counts()
    print(target_count)
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    values = {
        "Original":
        [target_count[positive_class], target_count[negative_class]]
    }

    df_positives = df[df[class_var] == positive_class]
    df_negatives = df[df[class_var] == negative_class]

    dfs = {}
    df_over = overSample(df_negatives, df_positives, values)
    df_over_X = df_over.drop([class_var], axis=1)
    df_over_y = df_over[class_var]
    df_smote = smote(df, class_var, values, negative_class, positive_class)
    df_smote_X = df_smote.drop([class_var], axis=1)
    df_smote_y = df_smote[class_var]
    dfs["over"] = (df_over_X, df_over_y)
    dfs["smote"] = (df_smote_X, df_smote_y)
    return dfs


# MULTICLASS -----------


def get_multiclass_results(y_true, y_pred):
    cf_mtrx = confusion_matrix(y_true, y_pred)
    scores = metrics.get_metrics(pd.DataFrame(cf_mtrx))
    avg_scores = np.mean(scores)
    return scores, avg_scores


def balancing_multiclass(X, y):
    over_sampler = SMOTE()
    return over_sampler.fit_resample(X, y)


def cross_validation(X, y, model, fold_count=5, multiclass=False, n_jobs=-1):
    metrics = ("accuracy", "balanced_accuracy", "recall", "precision", "f1")
    multiclass_metrics = ("balanced_accuracy", "recall_micro", "recall_macro",
                          "precision_micro", "precision_macro", "f1_micro",
                          "f1_macro")
    metrics_to_use = multiclass_metrics if multiclass else metrics
    score = cross_validate(model,
                           X,
                           y,
                           cv=fold_count,
                           scoring=metrics_to_use,
                           n_jobs=-1)
    score_indexes = ["test_" + metric for metric in metrics_to_use]
    metrics_results = itemgetter(*score_indexes)(score)
    all_avg_score_sum = 0
    for metric_result in metrics_results:
        all_avg_score_sum += np.mean(metric_result)

    all_avg_score = all_avg_score_sum / len(metrics_results)
    return score, all_avg_score
