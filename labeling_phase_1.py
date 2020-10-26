from utils import *

def rc_single(df, minutes):
    time = df['TimeStamp']
    RT = df['Diff Time']
    class_us = df['Class By User']
    change = 0
    index_start = 1
    while change <= 10:
        if RT[index_start] != RT[index_start - 1] and class_us[index_start] != class_us[index_start - 1]:
            change = change + 1
        index_start = index_start + 1

    time_start = con_data(time[index_start])
    minutes_added = datetime.timedelta(minutes=minutes)
    time_end = time_start + minutes_added

    i = index_start

    average_RT = datetime.timedelta(hours=0, minutes=0, seconds=0)

    while con_data(time[i]) < time_end:
        average_RT = average_RT + convert_to_time_delta(con_data(RT[i]))
        i = i + 1

    return average_RT / (i - index_start)


def add_label(threshold, df):
    labels = []
    for i in range(len(df["TimeStamp"])):
        # print(i)
        if df["Class By User"][i] != df["Original Class"][i] or convert_to_time_delta(
                con_data(df["Diff Time"][i])) >= threshold:
            labels.append('distracted')
        else:
            labels.append('attentive')

    df['label'] = labels

def create_dataset_threshold(dataSets):
    RC_average = [rc_single(df, 5) for df in dataSets]
    sum_c = datetime.timedelta(hours=0, minutes=0, seconds=0)
    for x in RC_average:
        sum_c = sum_c + x
    threshold = sum_c / len(RC_average)
    for i in range(len(dataSets)):
        add_label(threshold, dataSets[i])
        dataSets[i].to_csv("data_tes/res/" + str(i) + "th1_99s_.csv", index=False)



