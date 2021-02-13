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

    print("Directory lists : ", list_dir)
    list_dir.sort()
    print("Sorted Directory lists : ", list_dir)
    for l_dir in list_dir:
        list_sub_dir = os.listdir(f'.\\dataset\\MV137 Tosshullet\\{l_dir}')
        print("list_sub_dir     :   ", list_sub_dir[0])
        print("Directory names : ", l_dir)
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
        df["Day"] = df['Tid'].dt.day
        df["Hour"] = df['Tid'].dt.hour
        df["Minute"] = df['Tid'].dt.minute
        df['Week_Of_Year'] = df['Tid'].dt.week
        df['Day_Of_Week'] = df['Tid'].dt.dayofweek
        df['Is_Weekend'] = df['Day_Of_Week'].apply(lambda x: "1" if x == 6 or x == 7 else "0")
        df['Flow'] = df['Trender-Målt mengde, MV137']
        # print("df features	:	", df.columns)
        # df.drop(['Year'], axis=1, inplace=True)
        # print("dataset : \n", dataset)
        # print("df :", df)
        # print("df_minute : ", df_minute)
        # df_minute = df_minute.drop(columns=["Tid", "ms", "Trender-Målt mengde, MV137"], inplace=True)
        df_minute = df.drop(columns=['Tid', 'ms', 'Trender-Målt mengde, MV137'])
        # print("df_minute : ", df_minute)

        df.drop(['Minute'], axis=1, inplace=True)
        gf = df.groupby(['Month', 'Day', 'Hour']).agg({'Week_Of_Year': 'first',
                                                            'Day_Of_Week': 'first',
                                                            'Is_Weekend': 'first',
                                                            'Flow': 'mean'}).reset_index()
        df = gf
        # print("df after group by:", df)
        df = df[df['Flow'] != 0]
        df_minute = df_minute[df_minute['Flow'] != 0]
        df = df.dropna()
        df = df.drop_duplicates()
        # print("df column names : ", df.columns)
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
            df_minute.to_csv(f'{path}\\all_files_minute.csv', mode='a', header=True)
        else:
            df.to_csv(f'{path}\\both_files.csv', mode='a', header=False)
            df_minute.to_csv(f'{path}\\all_files_minute.csv', mode='a', header=False)
        i = i + 1

    df1 = pd.read_csv(f'{path}\\all_files_minute.csv')
    df1['Leak'] = df1['Flow'].apply(lambda x: "1" if x > 26 else "0")
    df1.drop(['Unnamed: 0'], inplace=True, axis=1)
    df1.to_csv(f'{path}\\all_files_minute_leak.csv', mode='a', header=True)
    # print("df1 : ", df1)
    df_3h = df1
    df1["min"] = df1['Flow']
    df1["max"] = df1['Flow']
    df1["mean"] = df1['Flow']
    df1["median"] = df1['Flow']
    df1["std"] = df1['Flow']

    df2 = df1.drop(['Week_Of_Year', 'Day_Of_Week', 'Is_Weekend'], axis=1)

    gf = df2.groupby(['Year', 'Month', 'Day', 'Hour']).agg({'Flow': 'sum',
                                                            'min': 'min',
                                                            'max': 'max',
                                                            'mean': 'mean',
                                                            'median': 'median',
                                                            'std': 'std',
                                                            'Leak': 'max'}).reset_index()
    # print("gf  : ", gf)
    gf = gf.dropna()
    gf.to_csv(f'{path}\\all_files_description_1h.csv', mode='w', header=True)

    temp = []
    i = 1
    d = 0
    while i <= df_3h.shape[0]:
        temp.append(d)
        if i % 180 == 0:
            d = d + 1
        i = i + 1

    df_3h['3_h'] = temp
    gf_3h = df_3h.groupby(['Year', 'Month', 'Day', '3_h']).agg({'Flow': 'sum',
                                                                'min': 'min',
                                                                'max': 'max',
                                                                'mean': 'mean',
                                                                'median': 'median',
                                                                'std': 'std',
                                                                'Leak': 'max'}).reset_index()
    gf_3h = gf_3h.dropna()
    gf_3h.to_csv(f'{path}\\all_files_description_3h.csv', mode='w', header=True)
