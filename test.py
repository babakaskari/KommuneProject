import pickle
import os
import sys

cur_dir = os.getcwd()
print(cur_dir)
path = ".\\dataset\\pickle"
list_dir = os.listdir(path)
print(list_dir)
file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
print(loaded_object)
a = loaded_object
print("x_train : ", a['x_train'])



