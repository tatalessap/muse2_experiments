import pandas as pd
import numpy as np
import os
from sklearn.decomposition import FastICA, PCA

"""
spectrum block average and return of new spectrum 
"""
def subsequently_binned(spec, len_axis, step, type_axis, start, num_elements):
    avg = []

    for i in range(start, len_axis-step, step):
        avg.append(np.mean(spec[i:i+num_elements-1, :], type_axis))

    avg = np.array(avg)

    return avg

"""
method for increasing a feature matrix (used to obtain matrix with all the resulting features from the channels)
"""
def add_matrix_to_matrix(matrix, sub_matrix, multiplier, last_index):

    if matrix.size == 0:
        matrix = np.zeros(((sub_matrix.shape[0] * multiplier), sub_matrix.shape[1]))

        matrix[0:sub_matrix.shape[0]] = sub_matrix

        last_index = sub_matrix.shape[0]
    else:
        matrix[last_index:last_index + sub_matrix.shape[0]] = sub_matrix
        last_index = last_index + sub_matrix.shape[0]

    return matrix, last_index

def check_folder(name_dir):
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)
        print("Directory ", name_dir, " Created ")
    else:
        print("Directory ", name_dir, " already exists")

def filter_with_ica( X ):
    #pca = PCA(0.90).fit(X)
    pca = PCA().fit(X)
    n_c = pca.n_components_
    #n_c = 2
    ica = FastICA(n_components=n_c)
    ica.fit(X)
    eeg_ica = ica.fit_transform(X)
    eeg_restored = ica.inverse_transform(eeg_ica)

    return eeg_restored




