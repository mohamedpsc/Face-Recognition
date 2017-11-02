#construct datasets
##TODO tupling
import logging
##non human datasets 
#TODO add in databasereader module
from os import listdir, path

import numpy as np
from scipy.misc import imread, imresize

import database_reader

logging.basicConfig(level=logging.INFO)
human_data = database_reader.load()
human_train_set = human_data[0]
NEW_RES = (92,112)
nonhuman_dir = "cat-dogs-dataset/train"
nonhuman_files = listdir(nonhuman_dir)
nonhuman_files = list(
    map(
        lambda x : path.join(nonhuman_dir, x),
        nonhuman_files
    )
)

## load nonhuman dataset same length as humans


def process_nonhuman(imgname):
    img = imread(imgname, mode="L")
    img = imresize(img, NEW_RES)
    return img.flatten()
#create list 
non_human_dataset = list(
    map(
        process_nonhuman,
        nonhuman_files[0:len(human_train_set)]
    )
)
#convert to ndarray to allow for matrix manipulations
non_human_dataset = np.asmatrix(non_human_dataset)

classhuman_mean = human_train_set.mean(axis=1)
classnonhuman_mean = non_human_dataset.mean(axis=1)
diff = (classhuman_mean - classnonhuman_mean) 
classscattern_matrix = diff.T * diff
in_class_scattermatrix = list()

def compute_in_scatter_matrix(data):
    class_scatter = np.zeros(data.shape[1])
    for c in data:
        #center data
        temp = c - c.meax(axis=1)
        class_scatter += np.cov(temp, rowvar=False)
    return class_scatter

def lda(inclass_scatter, between_class_scatter):
    if not path.exists("lda_bonus_vec.npy") or not path.exists("lda_bonus_values"):
        eig_values, eig_vectors = np.linalg.eigh(np.linalg.inv(between_class_scatter) * inclass_scatter)
        # reorder to descending
        eig_values = np.flip(eig_values, axis=1) 
        eig_vectors = np.flip(eig_vectors, axis=1)
        np.save("lda_bonus_vec.npy", eig_vectors)
        np.save("lda_bonus_val.npy",eig_values)
    else:
        eig_values = np.load("lda_bonus_vec.npy")
        eig_values = np.load("lda_bonus_val.npy")

    return eig_vectors[:,0]  #take one eigvector as discrimtory eig

def lda_classify_nonhuman(train, test,dis_vector):
    #TODO setup dataset propely 
    from sklearn.neighbors import KNeighborsClassifier as KNC 
    knc = KNC(n_neighbors=1)
    train.data = train.data * dis_vector
    test.data = test.data * dis_vector
    knc.fit(train.data, train.labels)
    return knc.score(test.data, test.labels)
    
if __name__ == "__main__":