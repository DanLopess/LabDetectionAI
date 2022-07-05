#!/usr/bin/python

import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from lib import utils
import warnings
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")


def print_default_arg_format():
    print(
        "Please run the command as: 'python TestMe.py <inputFile> <optionalOutputFile>'"
    )


def prepare_dataset(data, target_name):
    data.dropna(inplace=True)
    data['Date'] = data['Date'].astype(str) + ' ' + data['Time'].astype(str)
    data.drop(columns=['Time'], inplace=True)
    format = "%d/%m/%Y %H:%M:%S"
    data['Date'] = data['Date'].apply(
        lambda date: datetime.datetime.strptime(date, format))
    data['Hour'] = data['Date'].apply(lambda time: time.hour)
    data.drop('Date', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = utils.prepare_features(data, target_name)
    return data


def run_predictions(x: pd.DataFrame, y: pd.DataFrame,
                    sim: ctrl.ControlSystemSimulation):
    y_pred = []
    for _, values in x.iterrows():
        for col, val in zip(x.columns, values):
            sim.input[col] = val
        sim.compute()
        val_pred = sim.output['Persons']
        pred = True if val_pred is not None and val_pred > 0.5 else False  # if its higher than 0.5 then there are more than 2 persons
        y_pred.append(pred)
    return utils.get_results(y, y_pred, show=False)


def run_system(data, target_name, output):
    system: ctrl.ControlSystem = utils.load_system("best_system")
    sim = ctrl.ControlSystemSimulation(system, cache=False)
    y = data[target_name]
    X = data.drop([target_name], axis=1)
    acc, rec, prec, f1 = run_predictions(X, y, sim)
    print(f'Accuracy: {acc}')
    print(f'Recall: {rec}')
    print(f'Precision: {prec}')
    print(f'F1-Score: {f1}')


def run(filename, output=None):
    target = 'Persons'
    data = pd.read_csv(filename)
    data = prepare_dataset(data, target)
    run_system(data, target, output)
    if (output):
        plt.savefig(f'{output}/cf_mtrx.png')
        print(f"Confusion Matrix saved to: {output}/cf_mtrx.png")
    else:
        plt.show()


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