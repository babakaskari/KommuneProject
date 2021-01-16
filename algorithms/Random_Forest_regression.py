import pickle
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


cur_dir = os.getcwd()
print(cur_dir)
path = ".\\dataset\\pickle"
list_dir = os.listdir(path)
print(list_dir)
file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
print(loaded_object)
df = loaded_object
print("x_train : ", df['x_train'])
dataset = df['df']

dId = 57975963
year = 2019
month = 6
day = 21
hour = 13
computation_range = np.arange(1,5, 1)
# computation_range = [5, 10, 17, 35, 55, 78, 1002]
what_hour = np.arange(1,3, 1)
# ========================= filtering the dataset accoring to id, year, month, day, hour and weekend
print("dataset : ", dataset)
indexHour = dataset[(dataset['Year'] == year) & (dataset['Month']== month) &
(dataset["hour"] == hour) & (dataset["Day"] == day)].index
print("index hour   :   ", indexHour)
try:
    if(int(dataset.loc[indexHour[0]].Is_weekend) == 0):
        dataset = dataset.loc[dataset['Is_weekend'] == "0"]
        day_type = "Day is Not Weekend"
        # print("filtered datase : ", dataset)
    else:
        dataset = dataset.loc[dataset['Is_weekend'] == "1"]
        day_type = "Day is Weekend"
        # print("filtered datase : ", dataset)


    print("dataset : ", dataset)
    dId_list = dataset.DeviceId.unique()
    # print("dId_list : ", dId_list)

    # ===============================
    # print("dataset : ", dataset)
    y_dataset = dataset.Value
    x_dataset = dataset.drop(["Value"], axis=1)
    # print("y_dataset : ",y_dataset)
    # print("x_dataset : ", x_dataset)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

    # print("x_train : ", x_train)
    # print("y_train : ", y_train)
    # print("x_test : ", x_test)
    # print("y_test : ", y_test)

    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
    # ////////////////////////////////////////  hyperparameter tuning
    print(RandomForestClassifier().get_params())
    # params = hyperparameter_tuning.rfr_hyperparameter_tuner(clf, x_train, y_train)
    # clf.set_params(**params)
    # ////////////////////////////////
    clf.fit(x_train, y_train)

except Exception as e:
    print("there is an error", e)


