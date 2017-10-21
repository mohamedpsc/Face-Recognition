import os
from scipy.misc import imread


'''This Function Load Images dataset from a given directory into ndarray
Parameters:
-----------
    dir {string}: Path to dataset
    train_count {int}: Number of images to load for training 
    test_count {int}: Number of images to load for testing
Return:
    training_dataset{ List[ndarray] }:
    test_dataset{ List[ndarray] }:'''
def load(dir='orl_faces', train_count=7, test_count=3):
    try:
        training_dataset = []
        test_dataset = []
        for folder in os.listdir(dir):
            path = dir+'/'+folder
            if os.path.isdir(path):
                files = os.listdir(path)
                for i in range(0, train_count):
                    training_dataset.append(imread(path+ '/' +files[i]).flatten())
                for i in range(train_count, train_count + test_count):
                    test_dataset.append(imread(path + '/' + files[i]).flatten())
    except (NotADirectoryError, IOError, Exception) as e:
        print(e.strerror)
    else:
        return training_dataset, test_dataset
