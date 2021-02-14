import pandas as pd
import numpy as np
import XGB_Classifier_decision_tree
import Random_Forest_Classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("1 for 1 Hour \n3 for 3 Hour : ")
x = input("please select the file : ")

print("1 for no Date  \n2 for ohe Date : ")
x_data = input("please select the file : ")

print("1 for Random Forest Tree  \n2 for XGboost : ")
model = input("please select the file : ")
if x == '1':
    file_name = 'all_files_description_1h.csv'
else:
    file_name = 'all_files_description_3h.csv'

df = pd.read_csv(f'csv\\{file_name}', index_col=0)
# print("df : ", df)
x_dataset = df.drop(['Year', 'Month', 'Day', 'Hour'], axis=1)
y_dataset = df['Leak']
min_max_scaler = MinMaxScaler()
x_dataset = min_max_scaler.fit_transform(x_dataset)
# print(x_dataset)
# print(y_dataset)
# ############################### date ohe
if x_data == '2':
    df_date = df[['Year', 'Month', 'Day', 'Hour']]
    ohe = OneHotEncoder(sparse=False)
    df_date = ohe.fit_transform(df_date)
    x_dataset = np.concatenate((df_date, x_dataset), axis=1)


# ############################### date ohe
x_train, x_test, y_train, y_test = train_test_split(x_dataset,
                                                    y_dataset,
                                                    test_size=0.20,
                                                    shuffle=True,
                                                    random_state=42)
if model == '1':
    Random_Forest_Classifier.random_forest_tree(x_train, x_test, y_train, y_test)
else:
    XGB_Classifier_decision_tree.xgb_classifier(x_train, x_test, y_train, y_test)

