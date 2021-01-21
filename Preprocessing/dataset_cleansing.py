import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob


def data_cleaner():
    i = 0
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
        df["Year"] = df['Tid'].dt.year
        df["Month"] = df['Tid'].dt.month
        print("df[Month]", df["Month"])
        df["Day"] = df['Tid'].dt.day
        df["Hour"] = df['Tid'].dt.hour
        df["Minute"] = df['Tid'].dt.minute
        df['Week_Of_Year'] = df['Tid'].dt.week
        df['Day_Of_Week'] = df['Tid'].dt.dayofweek
        df['Is_Weekend'] = df['Day_Of_Week'].apply(lambda x: "1" if x == 6 else ("1" if x == 7 else "0"))
        df['Flow'] = df['Trender-Målt mengde, MV137']
        # print("df features	:	", df.columns)
        df.drop(['Year'], axis=1, inplace=True)
        # print("dataset : \n", dataset)
        # print("df :", df)
        df.drop(['Minute'], axis=1, inplace=True)
        gf = df.groupby(['Month', 'Day', 'Hour']).agg({'Week_Of_Year': 'first',
                                                            'Day_Of_Week': 'first',
                                                            'Is_Weekend': 'first',
                                                            'Flow': 'mean'}).reset_index()
        df = gf
        # print("df after group by:", df)
        df = df[df['Flow'] != 0]
        df = df.dropna()
        df = df.drop_duplicates()
        print("df column names : ", df.columns)
        # column_names = df.columns
        # print("df 	:	", df)
        # print("path : ", path)
        df.to_csv(f'{path}\\{l_dir}.csv')
        # file = open(f'{path}\\both_files.csv', 'a')
        # file.write(' , Month, Day, Hour, Week_Of_Year, Day_Of_Week, Is_Weekend, Flow')
        # file.close()
        df["Shifted_Mean_Hour"] = df["Flow"].shift(periods=1)
        df = df.dropna()

        if i == 0:
            df.to_csv(f'{path}\\both_files.csv', mode='a', header=True)
        else:
            df.to_csv(f'{path}\\both_files.csv', mode='a', header=False)
        i = i + 1


