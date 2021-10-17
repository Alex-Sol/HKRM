import pickle
import numpy as np
import sys
import os

dict_data = {}

def read_pkl(file):
    #file_path = 'data/VisualGenome/graph/' + file
    file_path = file
    print(file)
    f = open(file_path, 'rb')
    data = pickle.load(f)
    dict_data[file.split('.')[0]] = data
    #np.savetxt(file.split('.')[0] + '.txt', data, delimiter=",")
    print(1)

if __name__ == '__main__':
    # file_list = os.listdir('data/VisualGenome/graph')
    # for file in file_list:
    #     if file.split('.')[-1] != 'pkl':
    #         continue
    #     read_pkl(file)
    read_pkl("/home/zengli/HKRM_data/data/cache/DIOR_train_gt_roidb.pkl")

    print(1)



