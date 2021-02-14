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


def xgb_classifier(x_train, x_test, y_train, y_test):
    clf = xgb.sklearn.XGBClassifier(n_estimators=100, booster="gblinear")
    # params = hyperparameter_tuner.xgb_hyperparameter_tuner(clf, x_train, y_train)
    # clf.set_params(**params)
    clf.fit(x_train, y_train)
    # xgb_pred = clf.predict(x_test)

    evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test)



