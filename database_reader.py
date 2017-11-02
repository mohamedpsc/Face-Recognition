import os
from scipy.misc import imread
import numpy

def load(dir='orl_faces', train_count=5):
    '''This Function Load Images dataset from a given directory into numpy.matrix\n
        Args:
        -----
        :param dir: Path to dataset
        :type dir: String
        :param train_count: Number of images to load for training
        :type train_count: Int
        :param test_count: Number of images to load for testing
        :type test_count: Int
        Return:
        -------
        :return: training_dataset, test_dataset, training_labels, test_labels
        :rtype:  numpy.matrix, numpy.matrix, numpy.matrix, numpy.matrix'''
    try:
        training_dataset = []
        test_dataset = []
        training_label = []
        test_label = []
        for folder in os.listdir(dir):
            path = dir + '/' + folder
            if os.path.isdir(path):
                files = os.listdir(path)
                # assert (len(files) < train_count), "Number of required train samples is larger than the available samples"
                counter = 0
                for i in range(1, len(files), 2):
                    if counter < train_count:
                        training_dataset.append(imread(path + '/' + files[i]).flatten())
                        training_label.append(folder)
                        counter += 1
                    else:
                        test_dataset.append(imread(path + '/' + files[i]).flatten())
                        test_label.append(folder)
                for i in range(0, len(files), 2):
                    if counter < train_count:
                        training_dataset.append(imread(path + '/' + files[i]).flatten())
                        training_label.append(folder)
                        counter += 1
                    else:
                        test_dataset.append(imread(path + '/' + files[i]).flatten())
                        test_label.append(folder)
    except (Exception) as e:
        print(e.strerror)
    else:
        return [numpy.asmatrix(training_dataset), numpy.asmatrix(test_dataset), numpy.asmatrix(training_label), numpy.asmatrix(test_label)]


def load_non_human():
    from os.path import  join
    from os import listdir
    from itertools import islice
    animals_dir = "cat-dogs-dataset/train"
    animal_files = listdir(animals_dir)
    animals_files_abs = list(map(
        lambda fname : join(animals_dir, fname),
        animal_files
    ))

    animal_imgs = list(
        map(
            preprocessing_data,
            animals_files_abs[:200]
        )
    )

    animals_tests = list(
        map(
            preprocessing_data,
            animals_files_abs[200:400]
        )
    )
    train_labels = numpy.zeros([1,len(animal_imgs)])
    test_labels = numpy.zeros([1,len(animals_tests)])

    return  [numpy.asmatrix(animal_imgs), numpy.asmatrix(animals_tests), train_labels,test_labels]
    
import logging
logging.basicConfig(level=logging.INFO)
def preprocessing_data(imgname):
    '''gresacling and resizing images'''
    logging.debug(imgname)
    NEW_RES = (92,112)
    from scipy.misc import imread, imresize
    img = imread(imgname, mode="L")
    img = imresize(img, NEW_RES)
    return img.flatten()    


if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels = load(train_count=7)

