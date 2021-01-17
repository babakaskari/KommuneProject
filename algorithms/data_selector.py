import pickle
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import evaluator


def data_selector(clf, month, day, hour, hour_range, hour_from):
    cur_dir = os.getcwd()
    print(cur_dir)
    # print(os.path.basename(cur_dir)) # 'test.txt')
    path = os.path.dirname(cur_dir)
    print(os.path.dirname(cur_dir)) # '/test1/test2/test3'

    # print(os.path.basename(os.path.dirname(cur_dir))) # 'test3'

    path = f'{path}\\dataset\\pickle'
    print("path :", path)
    list_dir = os.listdir(path)
    print(list_dir)
    # file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
    file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    # print(loaded_object)
    df = loaded_object
    dataset = df['df']

    print("dataset : ", dataset)
    indexHour = dataset[(dataset['Month'] == month) &
    (dataset["Hour"] == hour) & (dataset["Day"] == day)].index
    # try:
    print("index hour   :   ", indexHour)
    print("index today  :   ", int(dataset.loc[indexHour[0]].Day_Of_Week))
    dow = int(dataset.loc[indexHour[0]].Day_Of_Week)

    dataset = dataset.loc[dataset['Day_Of_Week'] == dow]
    print("filterede dataset according to day : \n", dataset)

    # y_dataset = dataset.Flow
    #
    # x_dataset = dataset.drop(["Flow"], axis=1)
    x_dataset = dataset.drop(["Flow"], axis=1)
    y_dataset = dataset.loc[:, ["Flow"]]
    # ////////////////one hot encoding
    ohe = OneHotEncoder(sparse=False)
    x_dataset = ohe.fit_transform(x_dataset)
    # ////////////////////////////////
    # //////////////// normalization
    scaler = StandardScaler()
    # print(y_dataset)
    y_dataset = scaler.fit_transform(y_dataset)
    # /////////////////////////////////////////////////

    print("y_dataset : ", y_dataset)
    print("x_dataset : ", x_dataset)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2,
                                                        random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
    clf.fit(x_train, np.ravel(y_train))
    evaluation_dict = evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv, x_predict)

    # except Exception as e:
    #     print("there is an error, choose another date", e)

