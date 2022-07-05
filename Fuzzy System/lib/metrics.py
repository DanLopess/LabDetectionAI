import pandas as pd
import numpy as np


def get_tp(data):
    return np.trace(data)  # sum of diagonal


def get_tp_i(data, i):
    return data.iloc[i][i]  # element of diagonal


def get_tn_i(data, i):
    mini_matrix = data.drop(data.columns[[i]], axis=1)
    mini_matrix.drop(mini_matrix.index[i], axis=0, inplace=True)
    tn = mini_matrix.values.sum()
    return tn


def get_tn(data):
    tn_sum = 0
    for i in range(len(data)):
        tn_sum += get_tn_i(data, i)
    return tn_sum


def get_fp_i(data, i):
    return data.iloc[i].sum() - data.iloc[i][i]


def get_fp(data):
    fp_sum = 0
    for i in range(len(data)):
        fp_sum += get_fp_i(data, i)
    return fp_sum


def get_fn_i(data, i):
    # sum of column errors (error = all - valid)
    return data.iloc[:, i].sum() - data.iloc[i][i]


def get_fn(data):
    fn_sum = 0
    for i in range(len(data)):
        fn_sum += get_fn_i(data, i)
    return fn_sum


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def f_measure_m(pr, re):
    return 2 * pr * re / (pr + re)


def f_measure(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def micro_precision(data):
    tp = get_tp(data)
    fp = get_fp(data)
    pr_micro = precision(tp, fp)
    print(f"Micro Precision: {pr_micro}")
    return pr_micro


def macro_precision(data):
    pr_sum = 0
    for i in range(len(data)):
        tp = get_tp_i(data, i)
        fp = get_fp_i(data, i)
        pr_i = precision(tp, fp)
        pr_sum += pr_i
    pr_macro = pr_sum / len(data)
    print(f"Macro Precision: {pr_macro}")
    return pr_macro


def micro_recall(data):
    tp = get_tp(data)
    fn = get_fn(data)
    re_micro = recall(tp, fn)
    print(f"Micro Recall: {re_micro}")
    return re_micro


def macro_recall(data):
    re_sum = 0
    for i in range(len(data)):
        tp = get_tp_i(data, i)
        fn = get_fn_i(data, i)
        re_i = recall(tp, fn)
        re_sum += re_i
    re_macro = re_sum / len(data)
    print(f"Macro Recall: {re_macro}")
    return re_macro


def micro_accuracy(data):
    tp = get_tp(data)
    acc_micro = tp / data.values.sum()
    print(f"Micro Accuracy: {acc_micro}")
    return acc_micro


def macro_accuracy(data):
    acc_sum = 0
    for i in range(len(data)):
        tp = get_tp_i(data, i)
        tn = get_tn_i(data, i)
        fp = get_fp_i(data, i)
        fn = get_fn_i(data, i)
        acc_i = accuracy(tp, tn, fp, fn)
        acc_sum += acc_i
    acc_macro = acc_sum / len(data)
    print(f"Macro Accuracy: {acc_macro}")
    return acc_macro


def micro_f(data):
    tp = get_tp(data)
    fp = get_fp(data)
    fn = get_fn(data)
    f_micro = f_measure(tp, fp, fn)
    print(f"Micro F-measure: {f_micro}")
    return f_micro


def macro_f(data):
    f_sum = 0
    for i in range(len(data)):
        tp = get_tp_i(data, i)
        fp = get_fp_i(data, i)
        fn = get_fn_i(data, i)
        f_i = f_measure(tp, fp, fn)
        f_sum += f_i
    f_macro = f_sum / len(data)
    
    print(f"Macro F-measure: {f_macro}")
    return f_macro



def get_metrics(conf_mtrx):
    mi_pr = micro_precision(conf_mtrx)
    ma_pr = macro_precision(conf_mtrx)
    mi_re = micro_recall(conf_mtrx)
    ma_re = macro_recall(conf_mtrx)
    mi_ac = micro_accuracy(conf_mtrx)
    ma_ac = macro_accuracy(conf_mtrx)
    mi_f1 = micro_f(conf_mtrx)
    ma_f1 = macro_f(conf_mtrx)
    return  mi_pr, ma_pr, mi_re, ma_re, mi_ac, ma_ac, mi_f1, ma_f1
