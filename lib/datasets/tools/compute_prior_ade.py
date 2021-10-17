import numpy as np
import pickle
import os
import sys

NUM_ATTR_REL = 200
def cout_w(prob, num=NUM_ATTR_REL,dim=1):
    prob_weight = prob[:, :num]
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim) + 0.1
    temp = np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight

def cp_kl(a, b):
    # compute kl diverse
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 1
    sum_ = a * np.log(a / b)
    all_value = [x for x in sum_ if str(x) != 'nan' and str(x) != 'inf']
    kl = np.sum(all_value)
    return kl

def compute_js(attr_prob):
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))
    similarity[0, 1:] = 1
    similarity[1:, 0] = 1
    for i in range(1, cls_num):
        if i % 50 == 0:
            print('had proccessed {} cls...\n'.format(i))
        for j in range(1, cls_num):
            if i == j:
                similarity[i,j] = 0
            else:
                similarity[i,j] = 0.5 * (cp_kl(attr_prob[i, :], 0.5*(attr_prob[i, :] + attr_prob[j,:]))
                                         + cp_kl(attr_prob[j, :], 0.5*(attr_prob[i, :] + attr_prob[j, :])))
    return similarity

if __name__=='__main__':
    data_path = '../../../data/VisualGenome/graph/'
    dim_ = 445
    temp_pkl = pickle.loads(open(data_path + 'ade_graph_r.pkl', "rb").read())
    pickle.dump(temp_pkl, open(data_path + 'ade_graph_r_py2.pkl', "wb"), protocol=2)
    temp_pkl = pickle.loads(open(data_path + 'ade_graph_a.pkl', "rb").read())
    pickle.dump(temp_pkl, open(data_path + 'ade_graph_a_py2.pkl', "wb"), protocol=2)
    temp_pkl = pickle.loads(open(data_path + 'VOC_graph_r.pkl', "rb").read())
    pickle.dump(temp_pkl, open(data_path + 'VOC_graph_r_py2.pkl', "wb"), protocol=2)
    temp_pkl = pickle.loads(open(data_path + 'VOC_graph_a.pkl', "rb").read())
    pickle.dump(temp_pkl, open(data_path + 'VOC_graph_a_py2.pkl', "wb"), protocol=2)
