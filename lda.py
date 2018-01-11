import numpy as np
import database_reader as reader
from sklearn.neighbors import KNeighborsClassifier


def calculate_means(data, n_classes, spc):
    """
    Calculate mean for each data class and store it in a numpy matrix
    :param data: train data set
    :type data: array-like (nd-array, matrix, etc)
    :param n_classes: number of classes included in the given data set
    :type n_classes: int
    :param spc: number of samples given per class
    :type spc: int
    :return: computed means
    :rtype: numpy matrix
    """
    means = np.zeros((n_classes, data.shape[1]))
    for i in range(n_classes):
        means[i] = data[i * spc:i * spc + spc, :].mean(0)
    return means


def sb_matrix(means, n_classes, spc):
    """
    Calculate between classes scatter matrix
    :param means: classes mean
    :type means: array-like (nd-array, matrix, etc)
    :param n_classes: number of classes in the given data set
    :type n_classes: int
    :param spc: number of samples per class
    :type spc: int
    :return: between class scatter matrix
    :rtype: numpy matrix
    """
    all_class_mean = means.mean(0)
    b_matrix = np.zeros((means.shape[1], means.shape[1]))
    for i in range(n_classes):
        diff = (means[i, :] - all_class_mean)[:, np.newaxis]
        diff = spc * np.matmul(diff, diff.T)
        b_matrix += diff
    return b_matrix


def center_data(data, n_classes, spc, means=None):
    """
    Center train Data
    :param data: train data set
    :type data: array-like (nd-array, matrix, etc)
    :param n_classes: number of classes in the given data set
    :type n_classes: int
    :param spc: number of samples per class
    :type spc: int
    :param means: optional, classes means  (Default is None)
    :type means: array-like (nd-array, matrix, etc)
    :return: centered data
    :rtype: numpy matrix
    """
    index = 0
    if means is None:
        means = calculate_means(data, n_classes, spc)
    for i in range(n_classes):
        for j in range(0, spc):
            data[index] = data[index] - means[i]
            index += 1
    return data


def s_matrix(centered_data, n_classes, spc):
    """
    Calculate the within class scatter matrix
    :param centered_data: train data set after been centered.
    :param n_classes: number of classes in the given data set
    :param spc: number of samples per class
    :return: scatter matrix
    :rtype: numpy matrix
    """
    scatter_matrix = np.dot(centered_data[:spc, :].T, centered_data[:spc, :])
    for i in range(1, n_classes):
        scatter_matrix += np.dot(centered_data[i * spc:i * spc + spc, :].T, centered_data[i * spc:i * spc + spc, :])
    return scatter_matrix


def lda(data, n_classes, spc, save=False):
    """
    Perform linear discriminant analysis on the given train data set and return projection matrix
    :param data: train data set
    :type data: array-like (nd-array, matrix, etc)
    :param n_classes: number of classes included in the given data set
    :type n_classes: int
    :param spc: number of samples given per class
    :type spc: int
    :param save: if (True), it will store computed eigen values and vectors into .npy files
    :type save: bool
    :return: sorted eigen vectors used as projection matrix
    :rtype: numpy matrix
    """
    # Calculate classes means
    means = calculate_means(data, n_classes, spc)
    # Calculate the between class scatter matrix Sb
    b_matrix = sb_matrix(means, n_classes, spc)
    # Center data matrix
    # data matrix will be replaced by deviation matrix to save memory
    data = center_data(data, n_classes, spc, means)
    del means
    # calculate within class Scatter matrix (S)
    scatter_matrix = s_matrix(data, n_classes, spc)
    del data
    # Calculate eigen values and vectors
    eig_values, eig_vectors = np.linalg.eigh(np.dot(np.linalg.inv(scatter_matrix), b_matrix))
    del scatter_matrix
    del b_matrix
    # Sorting eigen values in descending order (needed only if eig used)
    # order = eigValues.argsort()[::-1]
    # eigValues = eigValues[order]
    # eigVectors = eigVectors[:, order]
    if save:
        np.save('lda_eigvector_'+str(spc), eig_vectors)
        np.save('lda_eigvalue_'+str(spc), eig_values)
    del eig_values
    return eig_vectors


def lda_classify(n_classes, spc, ndv=0, data_path='orl_faces', recompute=False):
    """
    :param n_classes: number of classes provided
    :type n_classes: int
    :param spc: number of samples per class
    :type spc: int
    :param ndv: number of dominant Vectors to be used to project data
    :type ndv: int
    :param data_path: path to data set
    :type data_path: String
    :param recompute: if (True), it will re compute projection matrix even if it already exists
    :type recompute: bool
    :return: accuracy of classification on the provided train data wrt test data
    :rtype: float
    """
    train_data, test_data, train_labels, test_labels = reader.load(dir=data_path, train_count=spc)
    from os import path
    if path.exists('lda_eigvector_'+str(spc)+'.npy') and not recompute:
        projection_matrix = np.load('lda_eigvector_'+str(spc)+'.npy')[::-1][:, :ndv]
    else:
        projection_matrix = lda(train_data, n_classes, spc, save=True)[::-1][:, :ndv]
    projected_data = test_data * projection_matrix
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_data * projection_matrix, train_labels.flat)
    return neigh.score(projected_data, test_labels.flat)


if __name__ == '__main__':
    for sample in [5, 7]:
        print(lda_classify(n_classes=40, spc=sample, ndv=39))
