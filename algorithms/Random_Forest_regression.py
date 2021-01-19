import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_selector
import visualizer


def random_forest_func():
    clf = RandomForestRegressor(max_depth=3, random_state=42, n_jobs=-1)
    result_file = data_selector.data_selector(clf)
    # print(result_file)
    visualizer.range_from_mae(result_file)
    visualizer.range_from_r2s(result_file)
    visualizer.range_from_mae_predicted(result_file)
    visualizer.range_from_r2s_predicted(result_file)
