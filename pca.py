import numpy


def deviationMatrix(matrix):
    ones = numpy.ones((1, matrix.shape[0]))
    devMatrix = matrix - ((ones * matrix)/matrix.shape[0])
    return devMatrix

def covariance(matrix):
    covMatrix = matrix.getT() * matrix
    return covMatrix / matrix.shape[0]

def pca(matrix, threshold, eigValues=None, eigVectors=None):
    # eigen Values and Vectors not calculated
    if eigValues is None or eigVectors is None:
        meanMatrix = numpy.mean(matrix, axis=0)
        devMatrix = deviationMatrix(matrix)
        covMatrix = covariance(devMatrix)
        eigValues, eigVectors = numpy.linalg.eigh(covMatrix)
    sum = eigValues.sum()
    num = 0
    count = 0
    while num/sum < threshold:
        num += eigValues[count]
        count += 1
    # Projecting Data on new Dimensions
    newData = matrix * eigVectors[:, count-1].T
    return newData

if __name__ == '__main__':
    import database_reader as reader
    train_data, test_data, train_labels, test_labels = reader.load()
    eigValues, eigVectors = pca(train_data, 0.1)
    print('Values\n', eigValues)
    print('Vectors\n', eigVectors)
