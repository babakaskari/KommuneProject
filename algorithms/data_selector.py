import pickle
import os
import sys
import numpy as np
import pandas as pd
import evaluator
import yaml
import hyperparameter_tuner
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def data_selector(clf):
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)

    month = int(param["initialize"]["month"])
    day = int(param["initialize"]["day"])
    hour = int(param["initialize"]["hour"])
    hour_range = np.arange(10, 15, 1)
    # hour_range = [5, 10, 17, 35, 55, 78, 1002]
    hour_from = np.arange(1, 5, 1)

    result = pd.DataFrame(
        columns=["Month", "Day", "Hour", "Hour Range", "Hour From", "Predicted Water Flow", "Mean Absolout Error",
                 "Mean Squared Error", "r2_score", "Explained Variance Regression Score", "Max Error"])
    cur_dir = os.getcwd()
    # print(cur_dir)
    # print(os.path.basename(cur_dir)) )
    path = os.path.dirname(cur_dir)
    # print(os.path.dirname(cur_dir))

    # print(os.path.basename(os.path.dirname(cur_dir))) #

    path_pickle = f'{path}\\dataset\\pickle'
    # print("path :", path_pickle)
    list_dir = os.listdir(path_pickle)
    # print(list_dir)

    # file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
    file_to_read = open(f'{path_pickle}\\{list_dir[0]}', "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    # print(loaded_object)
    df = loaded_object
    dataset = df['df']


    # ========================= filtering the dataset accoring to  day

    indexHour = dataset[(dataset['Month'] == month) & (dataset["Hour"] == hour) & (dataset["Day"] == day)].index
    # try:
    # print("index hour   :   ", indexHour)
    # print("index today  :   ", int(dataset.loc[indexHour[0]].Day_Of_Week))
    dow = int(dataset.loc[indexHour[0]].Day_Of_Week)
    # print("indexHour : ", indexHour[0])
    # print("dow : ", dow)
    dataset = dataset.loc[dataset['Day_Of_Week'] == dow].reset_index()
    # print("dataset : ", dataset)
    indexHour = dataset[(dataset['Month'] == month) &
                        (dataset["Hour"] == hour) &
                        (dataset["Day"] == day)].index
    # try:
    # print("index hour   :   ", indexHour[0])
    # print("filterede dataset according to day : \n", dataset)
    # y_dataset = dataset.Flow
    #
    i = 0
    for r in tqdm(hour_range):
        for h in tqdm(hour_from):
            # print("\n Hour Range : ", r, "  =====    Hour From : ", h)
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

            # print("y_dataset : ", y_dataset)
            # print("x_dataset : ", x_dataset)
            # print("y_dataset size: ", y_dataset.shape)
            # print("x_dataset size: ", x_dataset.shape)
            # print("indexHour : ", indexHour[0])
            x_predict = x_dataset[indexHour[0]]
            x_predict = x_predict.reshape(1, -1)
            y_predict = y_dataset[indexHour[0]]
            # print("Actual water consumption : ", y_predict[0])
            # y_predict = y_predict.to_frame()
            start_index = indexHour[0] - (r + h)
            end_index = indexHour[0] - h
            x_dataset = x_dataset[start_index: end_index]

            y_dataset = y_dataset[start_index: end_index]
            # print("x_predict : ", x_predict)
            # print("y_dataset : ", y_dataset)
            # print("x_dataset : ", x_dataset)

            x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2,
                                                                random_state=42)

            x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
            # ========================= hyperparameter tuning
            # params = hyperparameter_tuner.rfr_hyperparameter_tuner(clf, x_train, y_train)
            # clf.set_params(**params)
            # ===============================
            clf.fit(x_train, np.ravel(y_train))

            evaluation_dict = evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv, x_predict)
            # print(evaluation_dict)

            result.at[i, 'Month'] = month
            result.at[i, 'Day'] = day
            result.at[i, 'Hour'] = hour
            result.at[i, 'Hour Range'] = r
            result.at[i, 'Hour From'] = h
            result.at[i, 'Predicted Water Flow'] = evaluation_dict["predicted_value"][0]
            result.at[i, 'Mean Absolout Error'] = evaluation_dict["mean_absolute_error"]
            result.at[i, 'Mean Squared Error'] = pd.to_numeric(evaluation_dict["mean_squared_error"])
            result.at[i, "R2 score"] = evaluation_dict["r2_score"]
            result.at[i, 'Explained Variance Regression Score'] = evaluation_dict["explained_variance_regression_score"]
            result.at[i, 'Max Error'] = evaluation_dict["max_error"]
            i = i + 1

    path = os.path.dirname(cur_dir)
    # print(path)
    print("result: ", result)

    pd.to_numeric(result["Predicted Water Flow"])
    pd.to_numeric(result["Mean Squared Error"])
    pd.to_numeric(result["r2_score"])
    pd.to_numeric(result["Max Error"])
    pd.to_numeric(result["Explained Variance Regression Score"])
    pd.to_numeric(result["Mean Absolout Error"])
    least_mae_row_id = pd.to_numeric(result["Mean Absolout Error"]).idxmin(skipna=True)
    print(least_mae_row_id)
    print("Actual water flow : ", y_predict[0])
    print("Best predicted water flow : ", result.loc[least_mae_row_id]["Predicted Water Flow"])
    # print(result['Mean Absolout Error'].argmin())
    # print(list_dir[0])
    kernel_name = f'{clf}'.split("(", 1)
    file_name = list_dir[0].split(".", 1)[0]
    result.to_csv(f'{path}\\result\\{kernel_name[0]}_result_{file_name}.csv', index=False)
    return f'{kernel_name[0]}_result_{file_name}.csv'

