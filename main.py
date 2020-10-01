from classification import *
from generate_features import *

# default

# 1
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"

classifier = "weka.classifiers.trees.RandomForest"

path_packages = '/home/tatalessap/wekafiles/packages'

random = 3

# 2

# op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""

# classifier = "weka.classifiers.functions.SMO"

random = 3

# 4
# op=''

# classifier = "weka.classifiers.bayes.NaiveBayes"

# experiment1: feature by all files

# generate_by_all_files('res_eti_256row/', 'feature_ex')

path_file = 'feature_ex.csv'

path_folder = "one_file/"

#analyze_one_file(path_packages, path_file, path_folder, op, classifier, 5, random, "RandomForest")

# generate_by_one_file('res_eti_256row/', 'feature_ex')

path_folder = 'more_files/'

path_files = 'uni_features'

#analyze_files(path_packages, path_files, path_folder, 'byonefile', 5, op, classifier, random, "RandomForest")


# experiment2: seq
# generate_by_all_files('/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/', 'feature_ex')

path_folder = 'seq'

path_fe = 'feature_ex.csv'

path_ind = 'index_feature_ex.npz'

analyze_files_seq(path_packages, path_ind, path_fe, path_folder, op, classifier, '_byseq', 'RandomForest')
