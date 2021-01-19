import yaml
import random_forest_regression
import numpy as np

# load parameter yaml
with open("parameter.yaml") as stream:
    param = yaml.safe_load(stream)

month = int(param["initialize"]["month"])
day = int(param["initialize"]["day"])
hour = int(param["initialize"]["hour"])


print("Press 1 for RandomForestRegressor\nPress 2 for SVMregressor\n")
print('Please choose your model:    ')
selected_kernel = int(input())


if selected_kernel == 1:
    random_forest_regression.random_forest_func()
elif selected_kernel == 2:
    print("Ooooooooooops")
else:
    print("Exit")
    exit()



