import numpy as np

import database_reader

def LDA(train_data, nclasses, spc):
    """:parameter train_data: training Data set
    :type train_data: array-like (nd-array, matrix, etc)
    :parameter nclasses: number of classes included in the given dataset
    :type nclasses: int
    :parameter spc: Number of samples given per class
    :type spc: int"""
    # Calculate classes means
    means = np.zeros((nclasses, train_data.shape[1]))
    for i in range(0, nclasses):
        means[i] = train_data[i*spc:i*spc+spc, :].mean(0)
    all_class_mean = means.mean(0)
    # calculate the mu class Sb
    sb_matrix = np.zeros((train_data.shape[1], train_data.shape[1]))
    for i in range(0, nclasses):
        temp = np.subtract(means[i, :], all_class_mean)
        temp = spc * np.matmul(temp.T, temp)
        sb_matrix += temp
    # Calculate Deviation Matrix (Z)
    dev_matrix = np.zeros(train_data.shape)
    index = 0
    for i in range(0, nclasses):
        for j in range(0, spc):
            dev_matrix[index] = train_data[index] - means[i]
            index += 1
    # calculate Scatter matrix (S)
    scatter_matrix = np.dot(dev_matrix[:spc, :].T, dev_matrix[:spc, :])
    for i in range(nclasses):
        scatter_matrix += np.dot(dev_matrix[i*spc:i*spc+spc, :].T, dev_matrix[i*spc:i*spc+spc, :])
    # Calculate eigen values and vevtors
    eigValues, eigVectors = np.linalg.eigh(np.dot(np.linalg.pinv(scatter_matrix), sb_matrix))
    np.save('eigvalue', eigVectors)
    np.save('eigvector', eigValues)

def main():
    train_data, test_data, train_labels, test_labels = database_reader.load()
    LDA(train_data, 40, 5)


if __name__=='__main__':
    main()