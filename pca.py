import numpy
from sklearn.neighbors import KNeighborsClassifier
import database_reader as reader
import logging 
logging.basicConfig(level=logging.INFO)

def deviationMatrix(matrix):
    ones = numpy.ones((1, matrix.shape[0]))
    devMatrix = matrix - ((ones * matrix)/matrix.shape[0])
    return devMatrix

def covariance(matrix):
    covMatrix = matrix.getT() * matrix
    return covMatrix / matrix.shape[0]

def pca(matrix, threshold, eigValues=None, eigVectors=None):
    # eigen Values and Vectors Calculated For first Time
    if eigValues is None or eigVectors is None:
        devMatrix = deviationMatrix(matrix)
        covMatrix = covariance(devMatrix)
        eigValues, eigVectors = numpy.linalg.eigh(covMatrix)
        # Saving eigen Values and Vectors into Files
        numpy.save('eigValues', eigValues)
        numpy.save('eigVectors', eigVectors)
    # Finding Number Of Dimensions To Project Data on
    sum = eigValues.sum()
    num = 0
    count = eigValues.shape[1]
    while num/sum < threshold:
        count -= 1
        num += eigValues[0, count]
    # Return Projection Matrix
    return eigVectors[:, count:eigValues.shape[1]]

def classify(alpha):
    global train_data, test_data, train_labels, test_labels
    global eigValues, eigVectors
    projection_matrix = pca(train_data, alpha, eigValues, eigVectors)
    projected_data = test_data * projection_matrix
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_data * projection_matrix, train_labels.T)
    logging.debug(projected_data.shape)
    return neigh.score(projected_data, test_labels.T)

'''
loading constants 
'''
train_data, test_data, train_labels, test_labels = reader.load()
from os import path
if  path.exists('eigValues.npy') and path.exists('eigVectors.npy'):
            logging.info("Found eigen...loading...")
            eigValues = numpy.matrix(numpy.load('eigValues.npy'))
            eigVectors = numpy.matrix(numpy.load('eigVectors.npy'))
else:
        logging.info("Recomputing eigen..")
        
        proj_mat1 = pca(train_data, 0.8)
    

if __name__ == '__main__':
    from os import path
    from sys import argv

   
    proj_mat_08 = pca(train_data, 0.8, eigValues, eigVectors)
    proj_mat_085 = pca(train_data, 0.85, eigValues, eigVectors)
    proj_mat_09 = pca(train_data, 0.9, eigValues, eigVectors)
    proj_mat_095 = pca(train_data, 0.95, eigValues, eigVectors)

    # For alpha = 0.8
    for i in [0.8, 0.85, 0.9, 0.95]:
         print(classify(i))
