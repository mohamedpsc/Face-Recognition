import numpy as np
from numba import vectorize,cuda

import database_reader

'''This Function Load Images dataset from a given directory into ndarray
Parameters:
-----------
    train_data {matrix}: training examples dataset
    train_count {int}: Number of images to load for training 
    test_count {int}: Number of images to load for testing
Return:
    training_dataset{ List[ndarray] }:
    test_dataset{ List[ndarray] }:'''
def LDA(train_data):
    train_data_divided = np.vsplit(train_data, 40)
    class_mean = np.zeros((40, 10304), np.float64)
    # convert each sub matrix to a matrix and calculate each class mean
    for i in range(0, 40):
        class_mean[i] = np.asanyarray(train_data_divided[i]).mean(0)
    # calculate the total mean mu of all classes
    all_class_mean = class_mean.mean(0)
    # calculate the mu class Sb
    temp=np.zeros((40, 10304),np.float64)
    Sb_matrix= np.zeros((10304, 10304), np.float64)
    for i in range(0,40):
        temp=np.subtract(class_mean[i,:],all_class_mean)
        temp=5*np.matmul(temp.T,temp)
        Sb_matrix+=temp
    # prepare the mean class matrix
    temp=np.matrix
    class_mean_repeated = np.tile(class_mean[0], (5, 1))
    for i in range(1,40):
        # we will change this to handle different training sets
        temp = np.tile(class_mean[i], (5, 1))
        class_mean_repeated=np.append(class_mean_repeated,temp,0)
    # calculate Z centered matrix
    mu_=np.zeros((40, 10304),np.float64)
    # # we will change this limit if we  change number of train cases or test cases
    Z_matrix=np.subtract(train_data,class_mean_repeated)

    for i in range(0,200):
        temp=np.matmul(np.matrix.transpose(Z_matrix[i]),Z_matrix[i])

    print(S_Array,S_Array.shape)
    # print(S_Array)
    # for i in range(0,200):
      # S_matrix+=S_array[i]




def main():
    train_data, test_data, train_labels, test_labels = database_reader.load()
    LDA(train_data)


if __name__=='__main__':
    main()