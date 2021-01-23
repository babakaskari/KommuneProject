import pandas as pd
import numpy as np

import sklearn

from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_preds(model, x_train, y_train, x_true, y_true, x_cv, y_cv, x_predict):
    y_pred = model.predict(x_true)    
    pred = model.predict(x_predict)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    evr = explained_variance_score(y_true, y_pred)
    r2s = r2_score(y_true, y_pred)
    merror = max_error(y_true, y_pred)
    # print("y_pred : ", y_pred)
    # print("pred : ", pred)
    """
    print("pred : ", pred[0])
    print("Name of the kernel : ", model)
    print('Model Variance score: {}'.format(model.score(x_true, y_true)))
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print("explained variance regression score : ", evr)
    print("Max error : ", merror)
    print("R² score, the coefficient of determination  : ", r2s)
    """
    
    # ==================
    """
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=2)
    print("Cross validation score : ", cross_validation_score)
    cross_validation_predict = cross_val_predict(model, x_train, y_train, cv=2)
    # print("Cross validation predict : ", cross_validation_predict)
    cross_val_accuracy = np.mean(cross_validation_score) * 100
    print("cross validation accuracy : ", cross_val_accuracy)
    #return metric_dict
    """
    eval_dict = {
        "explained_variance_regression_score": evr,
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2s,
        "max_error": merror,
        "predicted_value": pred,

    }

    return eval_dict


