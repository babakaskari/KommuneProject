import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def file_reader(result_file):
    cur_dir = os.getcwd()
    path = os.path.dirname(cur_dir)
    # print(path)
    df = pd.read_csv(f'{path}\\result\\{result_file}')
    return df


def value_writer(plt, x_axis, y_axis):
    for x,y in zip(x_axis, y_axis):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center


def range_from_mae(result_file):
    df = file_reader(result_file)
    result = df.loc[df.groupby("Hour Range")["Mean Absolout Error"].idxmin()]
    computation_range  = result ['Hour Range'].tolist()
    what_hour = result ['Hour From'].tolist()
    mean_absolout_error = result ['Mean Absolout Error'].tolist()
    plt.plot(computation_range, what_hour, label='Hour From',  marker='o', linewidth=2)
    value_writer(plt, computation_range, what_hour)
    plt.plot(computation_range, mean_absolout_error, label='Mean Absolout Error', marker='o', linewidth=2)
    value_writer(plt, computation_range, mean_absolout_error)
    plt.xlabel('Hour Range')
    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title('MAE for Hour From and Hour Range')
    plt.show()


def range_from_r2s(result_file):
    df = file_reader(result_file)
    df = df.iloc[3:]
    result = df.loc[df.groupby("Hour Range")["R2 score"].idxmin()]
    hour_range  = result ['Hour Range'].tolist()
    hour_from = result ['Hour From'].tolist()
    r2s = result ['R2 score'].tolist()
    plt.plot(hour_range, hour_from, label='Hour From',  marker='o', linewidth=2)
    value_writer(plt, hour_range, hour_from)
    plt.plot(hour_range, r2s, label='R2 score', marker='o', linewidth=2)
    value_writer(plt, hour_range, r2s)
    plt.xlabel('Hour Range')
    plt.legend()
    plt.xticks(hour_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title('R2 score for Hour From and Hour Range')
    plt.show()


def range_from_mae_predicted(result_file):
    df = file_reader(result_file)
    result = df.loc[df.groupby("Hour Range")["Mean Absolout Error"].idxmin()]
    computation_range = result['Hour Range'].tolist()
    predicted = result['Predicted Water Flow'].tolist()
    what_hour = result['Hour From'].tolist()
    mean_absolout_error = result['Mean Absolout Error'].tolist()
    plt.plot(computation_range, what_hour, label='Hour From',  marker='o', linewidth=2)
    value_writer(plt, computation_range, what_hour)
    plt.plot(computation_range, predicted, label='Predicted Water Flow',  marker='o', linewidth=2)
    value_writer(plt, computation_range, predicted)
    plt.plot(computation_range, mean_absolout_error, label='Mean Absolout Error', marker='o', linewidth=2)
    value_writer(plt, computation_range, mean_absolout_error)
    plt.xlabel('Hour Range')
    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title('MAE for Hour From and Hour Range and Predicted FLow')
    plt.show()


def range_from_r2s_predicted(result_file):
    df = file_reader(result_file)
    df = df.iloc[3:]
    result = df.loc[df.groupby("Hour Range")["R2 score"].idxmin()]
    computation_range = result ['Hour Range'].tolist()
    predicted = result['Predicted Water Flow'].tolist()
    what_hour = result['Hour From'].tolist()
    r2s = result ['R2 score'].tolist()
    plt.plot(computation_range, what_hour, label='Hour From',  marker='o', linewidth=2)
    value_writer(plt, computation_range, what_hour)
    plt.plot(computation_range, predicted, label='Predicted Water Flow',  marker='o', linewidth=2)
    value_writer(plt, computation_range, predicted)
    plt.plot(computation_range, r2s, label='R2 score', marker='o', linewidth=2)
    value_writer(plt, computation_range, r2s)
    plt.xlabel('Hour Range')
    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title('R2 score for Hour From and Hour Range and Predicted Flow')
    plt.show()


