import os
from scipy.misc import imread,imresize
import numpy
NEW_RES = (50,50)
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
        def read_img(path, file_):
            p = os.path.join(path,file_)
            return imresize(
                imread(p),
                NEW_RES
            ).flatten()
        for folder in os.listdir(dir):
            path = dir + '/' + folder
            if os.path.isdir(path):
                files = os.listdir(path)
                # assert (len(files) < train_count), "Number of required train samples is larger than the available samples"
                counter = 0
                for i in range(1, len(files), 2):
                    if counter < train_count:
                        training_dataset.append(read_img(path,files[i]))
                        training_label.append(folder)
                        counter += 1
                    else:
                        test_dataset.append(read_img(path,files[i]))
                        test_label.append(folder)
                for i in range(0, len(files), 2):
                    if counter < train_count:
                        training_dataset.append(read_img(path,files[i]))
                        training_label.append(folder)
                        counter += 1
                    else:
                        test_dataset.append(read_img(path,files[i]))
                        test_label.append(folder)
    except (Exception) as e:
        print(e.strerror)
        print("err")
    else:
        return [numpy.asmatrix(training_dataset), numpy.asmatrix(test_dataset), numpy.asmatrix(training_label), numpy.asmatrix(test_label)]


def load_non_human(spc=10):
    from os.path import  join
    from os import listdir
    from itertools import islice
    # animals_dir = "cat-dogs-dataset/train"
    animals_dir = 'datasets/101_ObjectCategories/airplanes'
    animal_files = listdir(animals_dir)
    animals_files_abs = list(map(
        lambda fname : join(animals_dir, fname),
        animal_files
    ))
    animal_files_abs = list(
        filter(
            lambda x : "cat" in x,
            animals_files_abs
        )
    )

    animal_imgs = list(
        map(
            preprocessing_data,
            animals_files_abs[:spc]
        )
    )

    animals_tests = list(
        map(
            preprocessing_data,
            animals_files_abs[spc:spc+spc]
        )
    )

    train_labels = numpy.zeros([1,spc])
    test_labels = numpy.zeros([1,spc])
    ans = []
    ans.append(
        (
            numpy.asmatrix(animal_imgs),
            numpy.asmatrix(animals_tests),
            train_labels,
            test_labels
        )
    )

    h = load()
    ans.append(
        (
            h[0][0:spc,:],
            h[1][0:spc, :],
            numpy.ones([1,(spc)]),
            numpy.ones([1,(spc)])
        )
    )
    z = zip(ans[0], ans[1])
    ans = []
    for i,j in z:
        # print(i.shape,j.shape)
        # assert i.shape == j.shape
        ans.append(
            numpy.vstack((
                i,
                j
            ))
        )
    
    ans[2] = ans[2].flatten()
    ans[3] = ans[3].flatten()
    temp = ans[1] 
    ans[1] = ans[2] 
    ans[2] = temp
    del(temp)
    for i in ans:
        print(i.shape)
    return  iter(ans)

import logging
logging.basicConfig(level=logging.INFO)
def preprocessing_data(imgname):
    '''gresacling and resizing images'''
    logging.debug(imgname)
    from scipy.misc import imread, imresize
    img = imread(imgname, mode="L")
    img = imresize(img, NEW_RES)
    return img.flatten()


if __name__ == '__main__':
    # train_data, test_data, train_labels, test_labels = load(train_count=7)
    from pprint import pprint
    print(
        list(load_non_human())[3]
        )

