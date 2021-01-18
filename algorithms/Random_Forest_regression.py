import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_selector
import visualizer


month = 6
day = 21
hour = 13
hour_range = np.arange(10, 20, 1)
# hour_range = [5, 10, 17, 35, 55, 78, 1002]
hour_from = np.arange(1, 5, 1)

clf = RandomForestRegressor(max_depth=3, random_state=42, n_jobs=-1)
result_file = data_selector.data_selector(clf, month, day, hour, hour_range, hour_from)
# print(result_file)
visualizer.range_from_mae(result_file)
visualizer.range_from_r2s(result_file)
visualizer.range_from_mae_predicted(result_file)
visualizer.range_from_r2s_predicted(result_file)
