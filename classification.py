import weka
from weka.classifiers import Classifier, PredictionOutput, KernelClassifier, Kernel, Evaluation
import weka.core.converters as converters
from weka.core.classes import Random
import weka.core.jvm as jvm
import os
import pandas as pd
import numpy as np
from numpy import load


def analyze_files_seq(path_packages, path_ind, path_fe, path_folder, options, classifier, name_f, name):
    jvm.start(packages=path_packages)
    # jvm.start()

    ind_f = load(path_ind)

    lst = ind_f.files
    for item in lst:
        ind = ind_f[item] + 1

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    data = converters.load_any_file(path_fe)

    ind = np.append(ind, len(data))

    print(ind)

    data.class_is_last()

    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

    d_results = {'index': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    for j in range(len(ind) - 1):
        print(j)

        print(str(ind[j]) + '-' + str(ind[j + 1]))

        d_test = data.subset(row_range=str(ind[j]) + '-' + str(ind[j + 1]))

        if j == 0:  # first
            d_train = data.subset(row_range=str(ind[j + 1] + 1) + '-' + str(ind[-1]))  # last element
        elif j == len(ind) - 2:  # last
            print('ok')
            d_train = data.subset(row_range='1-' + str(ind[j] - 1))  # last element
        else:  # central
            s = '1-' + str(ind[j] - 1) + ',' + str(ind[j + 1] + 1) + '-' + str(ind[-1])
            d_train = data.subset(row_range=s)

        cls.build_classifier(d_train)

        evl = Evaluation(data)
        evl.test_model(cls, d_test, pout)

        save = pout.buffer_content()

        with open('predition/seq/' + name + str(j) + 'pred_data.csv', 'w') as f:
            f.write(save)

        d_results['index'].append(str(ind[j]))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    jvm.stop()

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder + '/' + name_f + name + 'results.csv', index=False)

    summary_result = {'type': [], 'mean percent correct': [], 'mean percent incorrect': []}

    summary_result['type'].append('mean')
    summary_result['mean percent correct'].append(d_results['percent_correct'].mean())
    summary_result['mean percent incorrect'].append(d_results['percent_incorrect'].mean())

    summary_result = pd.DataFrame(summary_result)

    summary_result.to_csv(path_folder + '/' + name_f + name + "summary.csv", index=False)  # save the fileve


def analyze_one_file(path_packages, path_file, path_folder, options, classifier, fold, random, name):
    print("start weka")
    # jvm.start()
    jvm.start(packages=path_packages)
    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))
    results = []
    d_results = {'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}
    data = converters.load_any_file(path_file)
    data.class_is_last()
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, fold, Random(random), pout)
    results.append(evl.summary())
    results.append("_______________________________________")
    d_results['percent_correct'].append(evl.percent_correct)
    d_results['percent_incorrect'].append(evl.percent_incorrect)
    d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    with open(path_folder + str(name) + "_results.txt", 'w') as f:
        for item in results:
            f.write("%s\n\n" % item)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder + str(classifier) + str(name) + '.csv', index=False)

    save = pout.buffer_content()

    with open('predition/all_file/pred_data' + str(name) + '.csv', 'w') as f:
        f.write(save)

    jvm.stop()


def analyze_files(path_packages, path_files, path_folder, name_file, fold, options, classifier, random, name):
    print("Class")

    jvm.start(packages=path_packages)
    # jvm.start()

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    results = []

    file_list = os.listdir(path_files)

    d_results = {'name_file': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    print(file_list)

    for file in file_list:
        print(str(file))
        data = converters.load_any_file(path_files + "/" + file)

        data.class_is_last()

        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

        evl = Evaluation(data)

        evl.crossvalidate_model(cls, data, fold, Random(random), pout)

        results.append(str(file) + ":")
        results.append(evl.summary())
        results.append("_______________________________________")

        # print(pout.buffer_content())

        d_results['name_file'].append(str(file))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

        save = pout.buffer_content()

        with open('predition/one_file/' + str(name) + str(file)[:-4] + 'pred_data.csv', 'w') as f:
            f.write(save)

    with open(path_folder + "/" + str(name) + "results.txt", 'w') as f:
        for item in results:
            f.write("%s\n\n" % item)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder + '/' + str(name) + str(classifier) + name_file, index=False)

    jvm.stop()
