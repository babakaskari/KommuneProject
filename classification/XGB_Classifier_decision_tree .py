import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
import pickle
import time
import os
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import sys
import evaluator
import hyperparameter_tuner
from sklearn.metrics import roc_curve
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from xgboost import XGBClassifier
# sys.path.insert(0, "C:\\Graphviz\\bin")
# sys.path.insert(0, "C:\\Graphviz")
# sns.set()

# file_to_read = open('.\\pickle\\preprocessed_dataset.pickle', "rb")
cur_dir = os.getcwd()
path = os.path.dirname(cur_dir)
path = f'{path}\\dataset\\pickle'
list_dir = os.listdir(path)
print(" file name : ", list_dir[-2])
print("path: ", path)
file_to_read = open(f'{path}\\data_all_files_description.pickle', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
dataset = loaded_object
print("dataset : ", dataset)

# clf = XGBClassifier()
y_dataset = dataset.loc[:, ["Leak"]]
x_dataset = dataset.drop(["Leak"], axis=1)
max_min_mean_median_std = x_dataset.drop(['Year', 'Month', 'Day'], axis=1)


print("x_dataset : ", x_dataset)
print("y_dataset : ", y_dataset)
x_train, x_test, y_train, y_test = train_test_split(max_min_mean_median_std,
                                                    y_dataset,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    stratify=y_dataset,
                                                    random_state=42)
# print("x_train : ", x_train)
# print("x_test : ", x_test)
# clf = xgb.sklearn.XGBClassifier(n_estimators=20,
#                                 max_depth=3,
#                                 learning_rate=0.3,
#                                 subsample=0.9,
#                                 colsample_bytree=1.0,
#                                 booster="gbtree",
#                                 eval_metric="error",
#                                 verbosity= 0,
#                                 n_jobs= -1)
clf = xgb.sklearn.XGBClassifier(n_estimators=100, booster="gblinear")
# params = hyperparameter_tuner.xgb_hyperparameter_tuner(clf, x_train, y_train)
# clf.set_params(**params)
clf.fit(x_train, y_train)
# xgb_pred = clf.predict(x_test)

evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test)

plt.figure(figsize=(20, 15))
xgb.plot_importance(clf, ax=plt.gca())
plt.show()
# plt.figure(figsize=(20, 15))
# xgb.plot_tree(clf, ax=plt.gca())
# plt.show()
# print("Number of boosting trees: {}".format(clf.n_estimators))
# print("Max depth of trees: {}".format(clf.max_depth))
# print("Objective function: {}".format(clf.objective))
# plot_tree(clf, num_trees=10)
plt.show()

