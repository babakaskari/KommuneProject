import pickle
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

    # print("x_train : ", x_train)
    # print("y_train : ", y_train)
    # print("x_test : ", x_test)
    # print("y_test : ", y_test)

    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
    # ////////////////////////////////////////  hyperparameter tuning
    print(RandomForestClassifier().get_params())
    # params = hyperparameter_tuning.rfr_hyperparameter_tuner(clf, x_train, y_train)
    # clf.set_params(**params)
    # ////////////////////////////////
    clf.fit(x_train, y_train)




