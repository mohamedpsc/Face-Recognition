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
    # convert each sub matrix to a matrix
    for i in range(0, 40):
        class_mean[i] = np.asanyarray(train_data_divided[i]).mean(0)
    all_class_mean = class_mean.mean(0)
    # print(class_mean, class_mean.shape)
    # print(all_class_mean, '\n', all_class_mean.shape)
    mu_class=np.zeros((40, 10304),np.float64)
    for i in range(0,40):
        mu_class[i]=np.subtract(class_mean[i,:],all_class_mean)

    Sb_matrix=np.matmul(np.matrix.transpose(mu_class),mu_class)
    Sb_matrix=40*Sb_matrix
    print(Sb_matrix,Sb_matrix.shape)





def main():
    train_data, test_data, train_labels, test_labels = database_reader.load()
    LDA(train_data)
if __name__=='__main__':
    main()