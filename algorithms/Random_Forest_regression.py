import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_selector_v1
from sklearn.model_selection import train_test_split

month = 6
day = 21
hour = 13
hour_range = np.arange(1, 5, 1)
# computation_range = [5, 10, 17, 35, 55, 78, 1002]
hour_from = np.arange(1, 3, 1)

# clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
clf = RandomForestRegressor(max_depth=3, random_state=42, n_jobs=-1)
data_selector_v1.data_selector(clf, month, day, hour, hour_range, hour_from)
# ////////////////////////////////////////  hyperparameter tuning
# print(RandomForestClassifier().get_params())
# params = hyperparameter_tuning.rfr_hyperparameter_tuner(clf, x_train, y_train)
# clf.set_params(**params)
# ////////////////////////////////

# clf.fit(x_train, np.ravel(y_train))







