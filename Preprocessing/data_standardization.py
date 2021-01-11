from tqdm import tqdm
import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_standard():
    # os.chdir('../')
    cur_dir = os.getcwd()
    # print(cur_dir)

    path = ("{dir}\\dataset\\csv".format(dir=cur_dir))
    list_dir = os.listdir(path)
    if not os.listdir(f'{path}'):
        print('The files are not present. Creating a new one..')
    else:
        print(list_dir)
        # print(list_dir[0])

    for file_list in list_dir:
        df = pd.read_csv(f'{path}\\{file_list}')

        # ohe = OneHotEncoder(sparse=False)
        # x_df_filtered_ohe = ohe.fit_transform(x_df_filtered)
        # print(df.columns)
        df.drop(['Unnamed: 0'], inplace=True, axis=1)
        # //////////// separating X and Y

        x_dataset = df.drop(["flow"], axis=1)
        y_dataset = df.loc[:, ["flow"]]

        # print(x_dataset)
        # ////////////////////////////////////////
        # ////////////////one hot encoding
        ohe = OneHotEncoder(sparse=False)
        x_dataset = ohe.fit_transform(x_dataset)
        # ////////////////////////////////
        # //////////////// normalization
        scaler = StandardScaler()
        # print(y_dataset)
        y_dataset = scaler.fit_transform(y_dataset)
        # /////////////////////////////////////////////////
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2,
                                                            random_state=42)

        x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

        result_dict = {

            "x_train": x_train,
            "x_test": x_test,
            "x_cv": x_cv,
            "y_train": y_test,
            "y_test": y_test,
            "y_cv": y_cv,
            "df": df,

        }

        f_t_write = open(f'{cur_dir}\\dataset\\pickle\\data_{os.path.splitext(file_list)[0]}.pickle', "wb")
        pickle.dump(result_dict, f_t_write)
        f_t_write.close()



