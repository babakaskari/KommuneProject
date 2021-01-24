import pickle
import os
import sys
import numpy as np
import pandas as pd
from xgboost import plot_tree
import xgboost as xgb
import sklearn
import evaluator_classifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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
file_to_read = open(f'{path_pickle}\\{list_dir[-2]}', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
df = loaded_object
dataset = df['df']
# print(dataset)

y_dataset = dataset.Leak
x_dataset = dataset.drop(['Leak'], axis=1)
# print(x_dataset)
# print(y_dataset)
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.3,
                                                                random_state=42)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

clf = xgb.sklearn.XGBClassifier(nthread=-1, n_estimators=50, seed=42)
clf.fit(x_train, y_train)
xgb_pred = clf.predict(x_train)
evaluator_classifier.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)
# plt.figure(figsize=(20, 15))
# xgb.plot_importance(clf, ax=plt.gca())
# plt.show()
# plt.figure(figsize=(20, 15))
xgb.plot_tree(clf, ax=plt.gca())
plt.show()
print("Number of boosting trees: {}".format(clf.n_estimators))
print("Max depth of trees: {}".format(clf.max_depth))
print("Objective function: {}".format(clf.objective))
plot_tree(clf, num_trees=10)
plt.show()

