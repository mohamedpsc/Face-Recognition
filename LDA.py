import numpy as np
from numba import vectorize,cuda

import database_reader

'''
Compute the d-dimensional mean vectors for the different classes from the dataset.
Compute the scatter matrices (in-between-class and within-class scatter matrix).
Compute the eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd) for the scatter matrices.
Sort the eigenvectors by decreasing eigenvalues and choose k
eigenvectors with the largest eigenvalues to form a d×k dimensional matrix WW
(where every column represents an eigenvector).
Use this d×k
eigenvector matrix to transform the samples onto the new subspace.
 This can be summarized by the matrix multiplication: 
 YY=XX×WW (where XX is a n×d-dimensional matrix representing the n samples, and yy are the transformed n×k-dimensional 
 samples in the new subspace).
'''
def LDA(train_data):
    train_data_divided = np.vsplit(train_data, 40)
    class_mean = np.zeros((40, 10304), np.float64)
    # convert each sub matrix to a matrix and calculate each class mean
    for i in range(0, 40):
        class_mean[i] = np.asanyarray(train_data_divided[i]).mean(0)
    # calculate the total mean mu of all classes
    all_class_mean = class_mean.mean(0)
    # calculate the mu class Sb
    mu_class=np.zeros((40, 10304),np.float64)
    for i in range(0,40):
        mu_class[i]=np.subtract(class_mean[i,:],all_class_mean)
    Sb_matrix=np.matmul(np.matrix.transpose(mu_class),mu_class)
    Sb_matrix=40*Sb_matrix
    # prepare the mean class matrix
    class_mean_repeated=np.matrix
    temp=np.matrix
    for i in range(0,40):
        # we will change this to handle different training sets
        temp = np.tile(class_mean[i], (5, 1))
        # temp=np.asarray(temp)
        class_mean_repeated=np.append(class_mean_repeated,temp,0)
    #     # np.append(class_mean_repeated,np.vstack(class_mean[i],5),axis=0)
    #     class_mean_repeated=np.concatenate((class_mean_repeated,temp))
    # class_mean_repeated=np.vstack(class_mean_repeated)
    print(class_mean_repeated,class_mean_repeated.shape)
    # # caculate Z centered matrix
    # mu_=np.zeros((40, 10304),np.float64)
    # # # we will change this limit if we  change number of train cases or test cases
    # Z_matrix=train_data-class_mean_repeated
    # for i in range(0,200):
    #   S_Array[i]=np.matmul(Z_matrix[i],np.matrix.transpose(Z_matrix[i]))
    #for i in range(0,200):
    #   S_matrix+=S_array[i]

def main():
    train_data, test_data, train_labels, test_labels = database_reader.load()
    LDA(train_data)
if __name__=='__main__':
    main()