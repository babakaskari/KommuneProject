from tqdm import tqdm
import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gaussrank import *
import matplotlib.pyplot as plt


def data_standard():
    # os.chdir('../')
    cur_dir = os.getcwd()
    print(cur_dir)

    path = ("{dir}\\dataset\\csv".format(dir=cur_dir))
    list_dir = os.listdir(path)
    if not os.listdir(f'{path}'):
        print('The files are not present. Creating a new one..')
    else:
        print(list_dir)
        # print(list_dir[0])

    for file_list in list_dir:
        df = pd.read_csv(f'{path}\\{file_list}')
        print("file name", file_list)
        # dataTypeSeries = df.dtypes
        # print('Data type of each column of Dataframe :')
        # print(dataTypeSeries)
        # ohe = OneHotEncoder(sparse=False)
        # x_df_filtered_ohe = ohe.fit_transform(x_df_filtered)
        # print(df.columns)
        # print("df : ", df)
        df.drop(['Unnamed: 0'], inplace=True, axis=1)
        # //////////// separating X and Y

        x_dataset = df.drop(["Flow"], axis=1)
        y_dataset = df.loc[:, ["Flow"]]

        # print(x_dataset)
        # ////////////////////////////////////////
        # ////////////////one hot encoding
        ohe = OneHotEncoder(sparse=False)
        x_dataset = ohe.fit_transform(x_dataset)
        # ////////////////////////////////
        # //////////////// normalization
        scaler = StandardScaler()
        # # print(y_dataset)
        y_dataset = scaler.fit_transform(y_dataset)

        # x_dataset = np.concatenate((x_dataset, z_dataset), axis=1)
        # print("x_dataset with shifte mean: ", x_dataset)

        # /////////////////////////////////////////////////
        ####################################################GAUSS RAN NORMALIZATION
        # x_cols = y_dataset.columns[:]
        # x = y_dataset[x_cols]
        # s = GaussRankScaler()
        # x_ = s.fit_transform(x)
        # assert x_.shape == x.shape
        # y_dataset[x_cols] = x_
        #
        # y_dataset = pd.DataFrame(y_dataset)
        # pd.plotting.scatter_matrix(y_dataset)
        # y_dataset.plot(kind='density', subplots=True, sharex=False)
        # plt.show()
        ####################################################
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2,
                                                            random_state=42)

        x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
        print("df columns :", df.columns)
        if "Shifted_Mean_Hour" in df:
            z_dataset = df.loc[:, ["Shifted_Mean_Hour"]]
            z_dataset = scaler.fit_transform(z_dataset)
            result_dict = {
                "x_dataset": x_dataset,
                "y_dataset": y_dataset,
                "z_dataset": z_dataset,
                "x_train": x_train,
                "x_test": x_test,
                "x_cv": x_cv,
                "y_train": y_test,
                "y_test": y_test,
                "y_cv": y_cv,
                "df": df,
            }
        elif "Leak" in df:
            x_df = df.drop(["Flow", "Leak"], axis=1)
            x_df = ohe.fit_transform(x_df)
            y_df = df.loc[:, ["Flow"]]
            y_leak = df.loc[:, ["Leak"]]
            y_df = scaler.fit_transform(y_df)
            df1 = pd.DataFrame(x_df)
            df1["y_df"] = y_df
            df1["Leak"] = y_leak
            result_dict = {
                "x_dataset": x_dataset,
                "y_dataset": y_dataset,
                "x_train": x_train,
                "x_test": x_test,
                "x_cv": x_cv,
                "y_train": y_test,
                "y_test": y_test,
                "y_cv": y_cv,
                "df": df1,

            }

        else:
            result_dict = {
                "x_dataset": x_dataset,
                "y_dataset": y_dataset,
                "x_train": x_train,
                "x_test": x_test,
                "x_cv": x_cv,
                "y_train": y_test,
                "y_test": y_test,
                "y_cv": y_cv,

        }

        f_t_write = open(f'{cur_dir}\\dataset\\pickle\\data_{os.path.splitext(file_list)[0]}.pickle', "wb")
        pickle.dump(result_dict, f_t_write)
        f_t_write.close()



