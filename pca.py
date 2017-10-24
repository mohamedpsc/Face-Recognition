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
        numpy.save('eigValues', eigValues)
        numpy.save('eigVectors', eigVectors)
    sum = eigValues.sum()
    num = 0
    count = eigValues.shape[1]-1
    while num/sum < threshold:
        num += eigValues[0, count]
        count -= 1
    # Projecting Data on new Dimensions
    newData = matrix * eigVectors[:, count:eigValues.shape[1]]
    return newData

if __name__ == '__main__':
    import database_reader as reader
    train_data, test_data, train_labels, test_labels = reader.load()
    try:
        eigValues = numpy.matrix(numpy.load('eigValues.npy'))
        eigVectors = numpy.matrix(numpy.load('eigVectors.npy'))
    except Exception as e:
        print('No Eigen Values or Vectors Found, Recomputing...')
        new_data = pca(train_data, 0.8)
    else:
        new_data = pca(train_data, 0.8, eigValues, eigVectors)
    print(new_data)
