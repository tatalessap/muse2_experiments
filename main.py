import os
from generate_features import *
from classification import *
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"

classifier = "weka.classifiers.trees.RandomForest"

path_packages = '/home/tatalessap/wekafiles/packages'

random = 3

#################
#generate_by_all_files('/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/', 'feature_ex')

#path_file = '/home/tatalessap/PycharmProjects/muse2_tesi/feature_ex.csv'

#analyze_one_file(path_packages, path_file, op, classifier, 5, random)

#################

#generate_by_all_files('/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/', 'feature_ex')

path_folder='/home/tatalessap/PycharmProjects/muse2_tesi/res_seq/'

path_file = '/home/tatalessap/PycharmProjects/muse2_tesi/feature_ex.csv'

path_ind = '/home/tatalessap/PycharmProjects/muse2_tesi/index_feature_ex.csv.npz'

path_fe = '/home/tatalessap/PycharmProjects/muse2_tesi/feature_ex.csv'

analyze_files_seq(path_packages, path_ind, path_fe, path_folder, op, classifier, '_byseq')




##################

"""
#generate_by_one_file('/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/', 'feature_ex')

path_folder ='/home/tatalessap/PycharmProjects/muse2_tesi/res_uni/'

path_files ='/home/tatalessap/PycharmProjects/muse2_tesi/uni_features/'

analyze_files(path_packages, path_files, path_folder, 'byonefile', 5, op, classifier, random)
"""






