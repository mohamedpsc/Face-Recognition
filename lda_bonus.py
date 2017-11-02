import logging
from os import listdir, path
import numpy as np
from scipy.misc import imread, imresize
import database_reader

logging.basicConfig(level=logging.INFO)

def compute_in_scatter_matrix(data):
        #center data
    logging.info("Start computing in cla")
    return data.T * data

def computer_between_class_scatter(class1data, class2data):
    logging.info("Start computing between class scatter matrix")
    m1 = class1data.mean(axis=0)
    m2 = class2data.mean(axis=0)
    diff = m1 - m2 
    return diff.T * diff


def lda(inclass_scatter, between_class_scatter):
    logging.info("start lda")
    if  path.exists("lda_bonus_vec.npy") and path.exists("lda_bonus_val.npy"):
        eig_vectors = np.load("lda_bonus_vec.npy")
        eig_values = np.load("lda_bonus_val.npy")
    else:
        # reorder to descending
        # eig_values = np.flip(eig_values, axis=1) 
        # eig_vectors = np.flip(eig_vectors, axis=1)
        logging.info("Recomputing eigens? press any key to continue")
        # input()
        eig_values, eig_vectors = np.linalg.eigh(np.linalg.pinv(inclass_scatter) * between_class_scatter)
        np.save("lda_bonus_vec.npy", eig_vectors)
        np.save("lda_bonus_val.npy",eig_values)

    return eig_vectors[:,-1]  #take one eigvector as discrimtory eig

def lda_classify_nonhuman(data, dis_vector):
    logging.info("Start classification")
    from sklearn.neighbors import KNeighborsClassifier as KNC 
    knc = KNC(n_neighbors=1)
    dis_vector = np.asmatrix(dis_vector)
    knc.fit(
        data[TRAIN_IDX] * dis_vector.T, 
        data[TRAIN_LABELS].flatten())
    return knc.score(data[TEST_IDX] * dis_vector.T, data[TEST_LABELS].flatten())

def concatenate_datasets(dataset1, dataset2):
    logging.info("Concatenating datasets")
    ans = list()
    for i,j in zip(dataset1, dataset2):
        ans.append(np.vstack((i,j)))
    return ans

TRAIN_IDX, TEST_IDX, TRAIN_LABELS, TEST_LABELS = [0, 1, 2, 3]
if __name__ == "__main__":
    import database_reader as dbr 
    humans = dbr.load()
    humans[2] = np.ones([1,len(humans[0])])
    humans[3] = np.ones([1,len(humans[1])])
    animals =dbr.load_non_human()

    inbetweenClass_scatter = computer_between_class_scatter(humans[0], animals[0])
    animals_scatter = compute_in_scatter_matrix(animals[0])
    humans_scatter = compute_in_scatter_matrix(humans[0])
    all_class_scatter = animals_scatter + humans_scatter
    print(humans_scatter.shape)
    print(animals_scatter.shape)

    dis_vector = lda(all_class_scatter, inbetweenClass_scatter)

    # dis_vector = lda(None, None)

    data = concatenate_datasets(humans, animals)
    print(lda_classify_nonhuman(data, dis_vector))
    
