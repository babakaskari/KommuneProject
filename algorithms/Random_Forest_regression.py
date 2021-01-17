import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_selector
from sklearn.model_selection import train_test_split

month = 6
day = 21
hour = 13
hour_range = np.arange(10, 12, 1)
# computation_range = [5, 10, 17, 35, 55, 78, 1002]
hour_from = np.arange(1, 5, 1)
clf = RandomForestRegressor(max_depth=3, random_state=42, n_jobs=-1)
data_selector.data_selector(clf, month, day, hour, hour_range, hour_from)










