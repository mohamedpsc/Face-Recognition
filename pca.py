import numpy
from sklearn.neighbors import KNeighborsClassifier
import database_reader as reader
import logging 
logging.basicConfig(level=logging.INFO)


def center_data(data):
    """
    Center the given data matrix
    :param data: train data set
    :type data: array-like(nd-array, matrix, etc)
    :return: centered data matrix
    :rtype: numpy matrix
    """
    ones = numpy.ones((1, data.shape[0]))
    dev_matrix = data - ((ones * data) / data.shape[0])
    return dev_matrix


def covariance(data):
    """
    Compute Covariance for the given data matrix
    :param data: train data set
    :type data: array-like(nd-array, matrix, etc)
    :return: covariance matrix
    :rtype: numpy matrix
    """
    cov_matrix = data.getT() * data
    return cov_matrix / data.shape[0]


def dominant_vectors(eig_values, threshold):
    """
    Calculate number of dominant vectors to consider for the given threshold
    :param eig_values: eigen values computed for train data set
    :type eig_values: nd-array
    :param threshold: ratio determine number of dominant vectors to consider, should be 0 < ratio < 1
    :type threshold: float
    :return: number of dominant vectors
    :rtype: int
    """
    total = eig_values.sum()
    num = 0
    # Rotating eigen values
    eig_values = eig_values[::-1]
    for count in range(0, eig_values.shape[0]):
        num += eig_values[count]
        if num / total > threshold:
            return count + 1
    return count


def pca(data, threshold, spc, save=False):
    """
    Perform principle component analysis on the given train data set
    :param data: train data set
    :type data: array-like(nd-arra, matrix, etc)
    :param threshold: ratio used to determine number of dominant vectors to consider
    :type threshold: float
    :param spc: number of samples per class
    :type spc: int
    :param save: Optional, if(True), it will save computed eigen values and vectors into np files
    :type save: bool
    :return: dominant vectors (Projection Matrix)
    :rtype: nd-array
    """
    # Center data
    data = center_data(data)
    # Compute covariance matrix
    cov_matrix = covariance(data)
    del data
    # Computing eigen values and vectors
    eig_values, eig_vectors = numpy.linalg.eigh(cov_matrix)
    del cov_matrix
    # Saving eigen Values and Vectors into Files
    if save:
        numpy.save('pca_eigvalues_'+str(spc), eig_values)
        numpy.save('pca_eigvectors_'+str(spc), eig_vectors)
    # Finding Number Of Dimensions To Project Data on
    ndv = dominant_vectors(eig_values, threshold)
    # Return Projection Matrix
    return eig_vectors[:, -ndv:]


def pca_classify(alpha, spc, data_path='orl_faces', recompute=False):
    """
    Classify test data using K Nearest Neighbours.
    :param alpha: ratio used to determine number of dominant vectors to consider
    :type alpha: float
    :param spc: number of samples per class
    :type spc: int
    :param data_path: path to data set
    :type data_path: String
    :param recompute: if (True), it will re compute projection matrix even if it already exists
    :type recompute: bool
    :return: accuracy of classification on the provided train data wrt test data
    :rtype: float
        """
    train_data, test_data, train_labels, test_labels = reader.load(dir=data_path, train_count=spc)
    from os import path
    if path.exists('pca_eigvector_' + str(spc) + '.npy') and path.exists('pca_eigvalue_' + str(spc) + '.npy')\
            and not recompute:
        ndv = dominant_vectors(numpy.load('pca_eigvalue_' + str(spc) + '.npy'), alpha)
        projection_matrix = numpy.load('pca_eigvector_' + str(spc) + '.npy')[:, -ndv:]
    else:
        projection_matrix = pca(train_data, alpha, spc, save=True)
    proj_test_data = test_data * projection_matrix
    proj_train_data = train_data * projection_matrix
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(proj_train_data, train_labels.flat)
    return neigh.score(proj_test_data, test_labels.flat)


if __name__ == '__main__':
    for samples in [5, 7]:
        for alpha in [0.8, 0.85, 0.9, 0.95]:
            print(pca_classify(alpha, samples))
