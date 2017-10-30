import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from numba import vectorize,cuda
import logging

import database_reader

def LDA(train_data):
    samples_no=5
    nclasses=40
    type=np.float64
    train_data_divided = np.vsplit(train_data, nclasses)
    class_mean = np.zeros((nclasses, 10304), type)
    # convert each sub matrix to a matrix and calculate each class mean
    for i in range(0, nclasses):
        class_mean[i] = np.asanyarray(train_data_divided[i]).mean(0)
    # calculate the total mean mu of all classes
    all_class_mean = class_mean.mean(0)
    # calculate the mu class Sb
    temp=np.zeros((nclasses, 10304),type)
    Sb_matrix= np.zeros((10304, 10304), type)
    for i in range(0,nclasses):
        temp=np.subtract(class_mean[i,:],all_class_mean)
        temp=5*np.matmul(temp.T,temp)
        Sb_matrix+=temp
    # prepare the mean class matrix
    class_mean_repeated = np.tile(class_mean[0], (5, 1))
    for i in range(1,nclasses):
        # we will change this to handle different training sets
        temp = np.tile(class_mean[i], (5,1))
        class_mean_repeated=np.append(class_mean_repeated,temp,0)
    # calculate Z centered matrix
    # we will change this limit if we  change number of train cases or test cases
    Z_matrix=np.subtract(train_data,class_mean_repeated)
    S_matrix=np.zeros((10304,10304),type)
    S_matrix = np.dot(Z_matrix[:5, :].T, Z_matrix[:samples_no, :])
    for i in range(1,nclasses):
        S_matrix += np.dot(Z_matrix[i*samples_no:i*samples_no+samples_no,:].T,Z_matrix[i*samples_no:i*samples_no+samples_no,:])
    #
    # for i in range(0,200,step=5):
    #     temp=np.matmul(np.matrix.transpose(Z_matrix[i:1+5,:]),Z_matrix[i])
    #     S_matrix+=temp
    temp=np.matmul(np.linalg.inv(S_matrix),Sb_matrix)
    W_eigvalue,V_eigvector=np.linalg.eigh(temp)
    projection_matrix=V_eigvector[:,10304-39:]
    return projection_matrix

def classify():
    global train_data, test_data, train_labels, test_labels
    global eigValues, eigVectors
    projection_matrix = LDA(train_data)
    projected_data = test_data * projection_matrix
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_data * projection_matrix, np.ravel(train_labels.T) )
    return neigh.score(projected_data, test_labels.T)

def main():
    global train_data, test_data, train_labels, test_labels
    global eigValues, eigVectors
    train_data, test_data, train_labels, test_labels = database_reader.load()
    # projection=LDA(train_data)
    x=classify()
    print(x)
if __name__=='__main__':
    main()

# means = np.zeros((nclasses, train_data.shape[1]))
# for i in range(0, nclasses):
#     means[i] = train_data[i * spc:i * spc + spc, 🙂.mean(0)
#     all_class_mean = means.mean(0)

