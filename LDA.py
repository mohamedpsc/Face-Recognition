import numpy as np
import database_reader as reader
from sklearn.neighbors import KNeighborsClassifier

def lda(data, nclasses, spc, ndv=None):
    """
    :param data: training Data set
    :type data: array-like (nd-array, matrix, etc)
    :param nclasses: number of classes included in the given dataset
    :type nclasses: int
    :param spc: Number of samples given per class
    :type spc: int
    :param ndv: Number of dominant vectors required
    :type ndv: int"""
    # Calculate classes means
    means = np.zeros((nclasses, data.shape[1]))
    for i in range(nclasses):
        means[i] = data[i * spc:i * spc + spc, :].mean(0)
    all_class_mean = means.mean(0)
    # calculate the between class scatter matrix Sb
    sb_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(nclasses):
        temp = np.subtract(means[i, :], all_class_mean)[:, np.newaxis]
        temp = spc * np.matmul(temp.T, temp)
        sb_matrix += temp
    # Calculate Deviation Matrix (Z)
    dev_matrix = np.zeros(data.shape)
    index = 0
    for i in range(nclasses):
        for j in range(0, spc):
            dev_matrix[index] = data[index] - means[i]
            index += 1
    # calculate within class Scatter matrix (S)
    scatter_matrix = np.dot(dev_matrix[:spc, :].T, dev_matrix[:spc, :])
    for i in range(1, nclasses):
        scatter_matrix += np.dot(dev_matrix[i*spc:i*spc+spc, :].T, dev_matrix[i*spc:i*spc+spc, :])
    # Calculate eigen values and vevtors
    eigValues, eigVectors = np.linalg.eigh(np.dot(np.linalg.pinv(scatter_matrix), sb_matrix))
    print()
    np.save('lda_eigvector_'+str(spc), eigVectors)
    np.save('lda_eigvalue_'+str(spc), eigValues)
    if ndv is not None:
        return eigVectors[:, :(ndv)]
    return eigVectors

def lda_classify(nclasses, spc, ndv=None, recompute=False):
    """
    :param nclasses: number of classes provided
    :type nclasses: int
    :param spc: number of samples per class
    :type spc: int
    :param ndv: number of dominant Vectors to be used to project data
    :type ndv: int
    :param recompute: if (True), it will re compute LDA even if projection matrix already exist
    :type recompute: bool
    :return: accuracy of LDA on the provided train data wrt test data
    """
    train_data, test_data, train_labels, test_labels = reader.load()
    from os import path
    if path.exists('lda_eigvector_'+str(spc)+'.npy') and not recompute:
        projection_matrix = np.matrix(np.load('lda_eigvector_'+str(spc)+'.npy'))[:, train_data.shape[1] - ndv:]
    else:
        projection_matrix = lda(train_data, nclasses, spc, ndv)
    projected_data = test_data * projection_matrix
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_data * projection_matrix, train_labels.T)
    return neigh.score(projected_data, test_labels.T)

def lda_classy_bonus(spc=100,ndv=1, recompute=False):
    from os import path 
    dl = (reader.load_non_human(spc))
    d = next(dl)
    if path.exists('lda_eigvector_'+str(spc)+'.npy') and not recompute:
        projection_matrix = np.matrix(np.load('lda_eigvector_'+str(spc)+'.npy'))[:, -ndv:]
    else:
        projection_matrix = lda(d, 2, spc,ndv)

    knc = KNeighborsClassifier(n_neighbors = 1)
    knc.fit(
        d * projection_matrix,
        next(dl)
    )
    del(d) 
    return knc.score(
        next(dl) * projection_matrix,
        next(dl)
    )

if __name__ == '__main__':
    # print(lda_classify(40, 5, 39))
    from sys import argv

    print(lda_classy_bonus(
        200,
        int(argv[1]),
        recompute=True
        ))
