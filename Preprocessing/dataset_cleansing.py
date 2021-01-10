import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob


def data_cleaner():
    os.chdir('../')
    cur_dir = os.getcwd()
    # print(cur_dir)
    if not os.path.isdir(f"{cur_dir}\\dataset\\csv"):
        print('The directory is not present. Creating a new one..')
        os.mkdir(f"{cur_dir}\\dataset\\csv")
    else:
        files = glob.glob(f"{cur_dir}\\dataset\\csv\\*.csv")

        for f in files:
            try:
                # f.unlink()
                os.remove(f)
            except OSError as e:
                print("no file exist ")
        # os.remove(f"{cur_dir}/csv/*.csv")

    path = ("{dir}\\dataset\\csv".format(dir=cur_dir))
    list_dir = os.listdir(".\\dataset\\MV137 Tosshullet")
    for l_dir in list_dir:
        list_sub_dir = os.listdir(f'.\\dataset\\MV137 Tosshullet\\{l_dir}')
        print("list_sub_dir     :   ", list_sub_dir[0])
        df = pd.read_excel (f'.\\dataset\\MV137 Tosshullet\\{l_dir}\\{list_sub_dir[0]}')
        column_names = df.columns
        df = pd.DataFrame(columns=column_names)

        for exl_file in tqdm(list_sub_dir):
            df1 = pd.read_excel (f'.\\dataset\\MV137 Tosshullet\\{l_dir}\\{exl_file}')
            # print("df1:	", df1)
            df = pd.concat([df, df1], axis=0, ignore_index=False)

        df['Tid'] = pd.to_datetime(df['Tid'], dayfirst=True)
        df["year"] = df['Tid'].dt.year
        df["month"] = df['Tid'].dt.month
        df["day"] = df['Tid'].dt.day
        df["hour"] = df['Tid'].dt.hour
        df["minute"] = df['Tid'].dt.minute
        df['week_of_year'] = df['Tid'].dt.week
        df['day_of_week'] = df['Tid'].dt.dayofweek
        df['flow'] = df['Trender-Målt mengde, MV137']
        # print("df features	:	", df.columns)
        df = df.dropna()
        df = df.drop_duplicates()
        df.drop(columns=column_names, inplace=True)
        # print("df 	:	", df)
        df.to_csv(f'{path}\\{l_dir}.csv')



