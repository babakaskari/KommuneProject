import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
import pickle
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
import evaluator
import hyperparameter_tuner
from sklearn.metrics import roc_curve
import seaborn as sns
# sys.path.insert(0, "C:\\Graphviz\\bin")
# sys.path.insert(0, "C:\\Graphviz")
# sns.set()


def random_forest_tree(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
    # params = hyperparameter_tuner.rfr_hyperparameter_tuner(clf, x_train, y_train)
    # clf.set_params(**params)
    clf.fit(x_train, y_train)
    evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test)


