from classification import *
from generate_features import *
# default
# 1
#op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
#classifier = "weka.classifiers.trees.RandomForest"

# 2
# op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
# classifier = "weka.classifiers.functions.SMO"

# 4
#op=''
#classifier = "weka.classifiers.bayes.NaiveBayes"

# 5
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'

##
path_packages = '/home/tatalessap/wekafiles/packages'

random = 3

#path_file = 'waves/one_file/all_each128.csv' #dim: 17913 attentive: 10266 / distracted: 7647

#path_folder = "waves/result_with_spect/"

# experiment1: feature by all files

generate_by_all_files('res_eti_256row/', 'feature_ex')

#experiment_file_random(path_packages, path_file, path_folder, op, classifier, 5, random, "RandomTree_wave")


# experiment2: feature by one file

# generate_by_one_file('res_eti_256row/', 'feature_ex')

path_folder = "waves/more_files/result_with_spect/"

path_files = 'waves/more_files'

random = 0

#experiment_more_file(path_packages, path_files, path_folder, '_byonefile_', 5, op, classifier, random, "RandomTree_waves_seq")


# experiment3: seq

# generate_by_all_files('/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/', 'feature_ex')

path_folder = 'waves/one_file/result_with_spect/seq'

path_fe = 'waves/one_file/all_each128.csv'

path_ind = 'waves/one_file/index_each128.npz'

#experiment_sequential_file(path_packages, path_ind, path_fe, path_folder, op, classifier, '_byseq', 'seq_wave_random_forest')
