import numpy
import pickle


def deviationMatrix(matrix):
    ones = numpy.ones((1, matrix.shape[0]))
    devMatrix = matrix - ((ones * matrix)/matrix.shape[0])
    return devMatrix

def covariance(matrix):
    covMatrix = matrix.getT() * matrix
    return covMatrix / matrix.shape[0]

def pca(matrix, threshold):
    meanMatrix = numpy.mean(matrix, axis=0)
    devMatrix = deviationMatrix(matrix)
    covMatrix = covariance(devMatrix)
    eigValues, eigVectors = numpy.linalg.eigh(covMatrix)
    return eigValues, eigVectors

if __name__ == '__main__':
    import database_reader as reader
    training, test = reader.load()
    eigValues, eigVectors = pca(training, 0.1)
    print('Values', eigValues, 'Vectors', eigVectors)
    # pickle.dump([eigValues, eigVectors], open('log.txt', 'wb'))
