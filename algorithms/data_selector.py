import pickle
import os
import sys
import numpy as np
import pandas as pd


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
print(loaded_object)
df = loaded_object
print("x_train : ", df['x_train'])
dataset = df['df']

dId = 57975963
year = 2019
month = 6
day = 21
hour = 13
computation_range = np.arange(1, 5, 1)
# computation_range = [5, 10, 17, 35, 55, 78, 1002]
what_hour = np.arange(1,3, 1)
# ========================= filtering the dataset accoring to id, year, month, day, hour and weekend
print("dataset : ", dataset)
indexHour = dataset[(dataset['Month'] == month) &
(dataset["Hour"] == hour) & (dataset["Day"] == day)].index
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

except Exception as e:
    print("there is an error", e)

