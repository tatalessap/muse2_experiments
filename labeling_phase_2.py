from utils import *

def fix_columns(column, minute_row):
    new_column = []
    j = 0
    while j + minute_row < len(column):
        count_at = 0
        count_dis = 0
        for i in range(j, j + minute_row):
            if column[i] == 'attentive':
                count_at = count_at + 1
            else:
                count_dis = count_dis + 1
        if count_at > count_dis:
            sub = ['attentive'] * int(minute_row)
        else:
            sub = ['distracted'] * int(minute_row)

        new_column.extend(sub)
        j = j + minute_row

    count_at = 0
    count_dis = 0

    for i in range(j, len(column)):
        if column[i] == 'attentive':
            count_at = count_at + 1
        else:
            count_dis = count_dis + 1
    if count_at > count_dis:
        sub = ['attentive'] * int(len(column) - j)
    else:
        sub = ['distracted'] * int(len(column) - j)

    new_column.extend(sub)

    return new_column

def apply_to_datasets(path_csv, minute_row, path_save):
    dataSets = create_list_dataset(path_csv)
    for i in range(len(dataSets)):
        dataSets[i]['label'] = fix_columns(dataSets[i]['label'], minute_row)
    for i in range(len(dataSets)):
        dataSets[i].to_csv(path_save + str(i) + "with_256_row.csv", index=False)

