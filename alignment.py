import pandas as pd
import datetime
from utils import *


def fix_interface_dataset(path_ann):
    ann = pd.read_csv(path_ann, index_col=False)
    ann = ann.transpose()[1:].transpose()
    ann_drop = ann.dropna()
    first_check = ann_drop['check time'][1][:-3]  # considero come primo check la prima etichetta
    ann_drop = ann_drop[:-1]  # taglio ultima riga
    ann_drop = ann_drop[2:]  # parto dalla riga 2
    ann_drop['check time'] = list(map(lambda x: x[:-3], ann_drop['check time']))  # normalizzo il tempo
    return ann_drop, first_check

def fix_muse_dataset(path_muse):
    muse = pd.read_csv(path_muse)
    muse_slice = muse.transpose()[:25].transpose()  # drop columns che non servono
    muse_slice['TimeStamp'] = list(map(lambda x: x[11:], muse_slice['TimeStamp']))  # normalizzo tempo
    muse_tiny = muse_slice.dropna()  # elimino righe bianche
    muse_tiny = muse_tiny[(muse_tiny == 0).sum(1) < 8]  # elimino gruppi di righe bianche
    return muse_tiny

def alignment(ann_drop, muse_tiny, first_check):
    i = 0
    while con_data(first_check) > con_data(list(muse_tiny['TimeStamp'])[i]):
        i = i + 1
    index = i
    muse_tiny = muse_tiny[index:]
    print(index)
    j = 0
    eti = list(ann_drop['check time'])
    muse = list(muse_tiny['TimeStamp'])
    class_user = list(ann_drop['class by user'])
    class_ori = list(ann_drop['original class'])
    class_by_user = []
    original_class = []
    time_check = []
    time_diff = []
    for i in range(len(muse)):
        if j < len(eti) - 1:
            #if (con_data(muse[i]) > con_data(eti[j]) + ok):
            if con_data(muse[i])>con_data(eti[j]):
                j = j + 1
            if j == 0:
                time_diff.append(str(con_data(eti[j]) - con_data(first_check)))
            else:
                time_diff.append(str(con_data(eti[j]) - con_data(eti[j - 1])))
            class_by_user.append(class_user[j])
            original_class.append(class_ori[j])
            time_check.append(con_data(eti[j]).time())
            #time_check.append((con_data(eti[j]) + ok).time())
    muse_tiny_new = muse_tiny[:len(class_by_user)]
    muse_tiny_new["Class By User"] = class_by_user
    muse_tiny_new["Original Class"] = original_class
    muse_tiny_new["Time Check Button"] = time_check
    muse_tiny_new["Diff Time"] = time_diff

def save_csv(muse_tiny_new, name):
    muse_tiny_new.to_csv(name, index=False)





