import os
from scipy.misc import imread
import numpy


'''This Function Load Images dataset from a given directory into ndarray
Parameters:
-----------
    dir {string}: Path to dataset
    train_count {int}: Number of images to load for training 
    test_count {int}: Number of images to load for testing
Return:
    training_dataset{ List[ndarray] }:
    test_dataset{ List[ndarray] }:'''
def load(dir='orl_faces', train_count=3, test_count=7):
    try:
        training_dataset = []
        test_dataset = []
        training_label = []
        test_label = []
        for folder in os.listdir(dir):
            path = dir + '/' + folder
            if os.path.isdir(path):
                files = os.listdir(path)
                for i in range(0, train_count):
                    training_dataset.append(imread(path + '/' + files[i]).flatten())
                    training_label.append(folder)
                for i in range(train_count, train_count + test_count):
                    test_dataset.append(imread(path + '/' + files[i]).flatten())
                    test_label.append(folder)
    except (NotADirectoryError, IOError, Exception) as e:
        print(e.strerror)
    else:
        return numpy.asmatrix(training_dataset), numpy.asmatrix(test_dataset), numpy.asmatrix(training_label), numpy.asmatrix(test_label)

if __name__ == '__main__':
    training, test = load()
