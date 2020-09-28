import weka
from weka.classifiers import Classifier, PredictionOutput, KernelClassifier, Kernel, Evaluation
import weka.core.converters as converters
from weka.core.classes import Random
import weka.core.jvm as jvm
import os
import pandas as pd
import numpy as np
from numpy import load


def analyze_files_seq(path_packages, path_ind, path_fe, path_folder, options, classifier, name_f):

    jvm.start(packages=path_packages)

    ind_f = load(path_ind)

    lst = ind_f.files
    for item in lst:
        ind = ind_f[item] + 1

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    data = converters.load_any_file(path_fe)

    ind = np.append(ind, len(data))

    print(ind)

    data.class_is_last()

    d_results = {'index': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    for j in range(len(ind)-1):

        print(j)

        print(str(ind[j])+'-'+str(ind[j+1]))

        d_test = data.subset(row_range=str(ind[j]) + '-' + str(ind[j+1]))

        if j == 0:            # first
            d_train = data.subset(row_range=str(ind[j+1]+1)+'-'+str(ind[-1])) #last element
        elif j == len(ind)-2:   # last
            print('ok')
            d_train = data.subset(row_range='1-' + str(ind[j]-1))  # last element
        else:   # central
            s = '1-' + str(ind[j] - 1) + ',' + str(ind[j+1]+1)+'-'+str(ind[-1])
            d_train = data.subset(row_range=s)

        cls.build_classifier(d_train)

        evl = Evaluation(data)
        evl.test_model(cls, d_test)

        d_results['index'].append(str(ind[j]))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix()) #Generates the confusion matrix.

    jvm.stop()

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder + name_f +'results.csv', index=False)

    summary_result = {'type': [], 'mean percent correct': [], 'mean percent incorrect': []}

    summary_result['type'].append('mean')
    summary_result['mean percent correct'].append(d_results['percent_correct'].mean())
    summary_result['mean percent incorrect'].append(d_results['percent_incorrect'].mean())

    summary_result = pd.DataFrame(summary_result)

    summary_result.to_csv(path_folder+name_f+"summary.csv", index=False)  # save the fileve

def analyze_one_file(path_packages, path_file, path_folder, options, classifier, fold, random, i):
    print("Class")
    jvm.start(packages=path_packages)
    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))
    results = []
    d_results = {'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}
    data = converters.load_any_file(path_file)
    data.class_is_last()
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, fold, Random(random), pout)
    print(evl.summary())

    print("pctCorrect: " + str(evl.percent_correct))
    print("pctIncorrect: " + str(evl.percent_incorrect))

    results.append(evl.summary())
    results.append("_______________________________________")
    d_results['percent_correct'].append(evl.percent_correct)
    d_results['percent_incorrect'].append(evl.percent_incorrect)
    d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    with open(str(i)+"_results.txt", 'w') as f:
        for item in results:
            f.write("%s\n\n" % item)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder+str(classifier)+"with_name_file", index=False)

    jvm.stop()


def analyze_files(path_packages, path_files, path_folder, name_file, fold, options, classifier, random):

    print("Class")

    jvm.start(packages=path_packages)

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    results = []

    file_list = os.listdir(path_files)

    d_results = {'name_file': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    for file in file_list:
        print(str(file))
        data = converters.load_any_file(path_files + "/" + file)
        data.class_is_last()

        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
        evl = Evaluation(data)

        evl.crossvalidate_model(cls, data, fold, Random(random), pout)
        print(evl.summary())

        print(str(file))
        print("pctCorrect: " + str(evl.percent_correct))
        print("pctIncorrect: " + str(evl.percent_incorrect))

        results.append(str(file)+":")
        results.append(evl.summary())
        results.append("_______________________________________")

        # print(pout.buffer_content())

        d_results['name_file'].append(str(file))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

        """
        text_file = open(path_folder+"/buffers/" + str(file) + "_buffer.txt", "w")
                n = text_file.write(pout.buffer_content())
                text_file.close()
        """

    with open(path_folder+"/results.txt", 'w') as f:
        for item in results:
            f.write("%s\n\n" % item)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder+'/'+str(classifier)+name_file, index=False)

    jvm.stop()
