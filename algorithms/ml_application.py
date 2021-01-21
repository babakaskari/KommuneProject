import yaml
import random_forest_regression
import k_means_saved

# load parameter yaml
with open("parameter.yaml") as stream:
    param = yaml.safe_load(stream)

month = int(param["initialize"]["month"])
day = int(param["initialize"]["day"])
hour = int(param["initialize"]["hour"])

print("Press 0 for data preprocessing\nPress 1 for RandomForestRegressor\nPress 2 for k_means\n")
print('Please choose your model:    ')
selected_kernel = int(input())

if selected_kernel == 1:
    random_forest_regression.random_forest_func()
elif selected_kernel == 2:
    k_means_saved.k_means_model()
else:
    print("Exit")
    exit()



