import pickle
import os
import sys

cur_dir = os.getcwd()
print(cur_dir)

path = os.path.dirname(cur_dir)
path = f'{path}\\dataset\\pickle'
list_dir = os.listdir(path)
print(list_dir[0])




