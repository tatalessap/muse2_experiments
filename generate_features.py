import matplotlib.pyplot as plt
from utils import *


#/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/
def generate_by_all_files(path_file_csv, name_file):
    vector_classes_big = np.array([])
    last_index_big_matrix = 0
    complete_matrix = []
    dic = {}
    # the size of complete_matrix changes
    dim0 = 0
    files = os.listdir(path_file_csv)
    dataSets = [pd.read_csv(path_file_csv + str(el)) for el in files]
    index_file = 0
    indexes = []

    for df in dataSets:
        df = df[['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'label']]

        listCh = list(df.columns[:-1])

        # if ica: df = filter_with_ica(df)

        complete_matrix, vector_classes = get_complete_matrix(listCh, df)

        # save the complete_matrix to merge them all together at the end of the cycle
        dic[index_file] = np.transpose(complete_matrix)

        # update number of row of the big_matrix
        dim0 = dim0 + complete_matrix.shape[1]

        if vector_classes_big.shape[0] == 0:
            vector_classes_big = vector_classes
        else:
            vector_classes_big = np.concatenate((vector_classes_big, vector_classes))

        index_file = index_file + 1

    big_matrix = np.zeros((dim0, complete_matrix.shape[0]))

    print(complete_matrix.shape[0])

    # merge of complete_matrix
    for key in dic.keys():
        indexes.append(last_index_big_matrix)
        big_matrix[last_index_big_matrix:last_index_big_matrix + dic[key].shape[0]] = dic[key]
        last_index_big_matrix = last_index_big_matrix + dic[key].shape[0]

    dic.clear()

    big_matrix = pd.DataFrame(big_matrix)

    big_matrix['classes'] = vector_classes_big

    big_matrix.to_csv(name_file + '.csv', index=False)  # save the file

    np.savez('index_' + name_file+'.csv', np.array(indexes))

#/home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/
def generate_by_one_file(path_file_csv, name_file):
    bb = []
    dic = {}
    # the size of complete_matrix changes
    dim0 = 0
    files = os.listdir(path_file_csv)
    dataSets = [pd.read_csv(path_file_csv + str(el)) for el in files]
    index_file = 0

    for df in dataSets:
        index_file = 0
        df = df[['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'label']]

        listCh = list(df.columns[:-1])

        # if ica: df = filter_with_ica(df)

        complete_matrix, vector_classes = get_complete_matrix(listCh, df)

        big_matrix = pd.DataFrame(np.transpose(complete_matrix))

        big_matrix['classes'] = list(vector_classes)

        bb.append(big_matrix)

    index_file = 0
    for big_matrix in bb:
        big_matrix.to_csv('/home/tatalessap/PycharmProjects/muse2_tesi/uni_features/'+ name_file + str(index_file) + '.csv', index=False)  # save the file
        index_file = index_file + 1

"""
the complete_matrix is the matrix where is the merge of all feature_matrix. In this function there is the calculate of 
spectrogram of each task
"""
def get_complete_matrix(list_ch, data):

    last_index = 0

    complete_matrix = np.array([])

    vector_classes = np.array([])

    dfs = [x for _, x in data.groupby('label')]

    data_attentive = dfs[0]
    data_distracted = dfs[1]

    for el in list_ch:
        x = np.array(data_attentive[el])

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)



        # do the spectrogram
        # sP the spectrogram
        # freq, the vector of the frequencies
        # t, the vector of the time

        # first
        i = 0
        sP, freq, t, im = ax1.specgram(x, Fs=256, window=np.blackman(M=256), NFFT=256)

        spec_a, freq_ = steps(sP, freq)

        # second

        x = np.array(data_distracted[el])

        sP, freq, t, im = ax2.specgram(x, Fs=256, window=np.blackman(M=256), NFFT=256)
        spec_d, freq_ = steps(sP, freq)

        #
        feature_matrix = get_matrix_feature(spec_a, spec_d)

        plt.close(fig)

        #sP, freq, t, im = ax3.specgram(feature_matrix)

        #plt.show()

        vector_classes = get_classes(spec_a.shape[1], spec_d.shape[1])

        # merge of feature_matrix of the single channel
        complete_matrix, last_index = add_matrix_to_matrix(complete_matrix, feature_matrix, len(list_ch), last_index)

    return complete_matrix, vector_classes


def steps(sP, freq):
    """
    These were subsequently binned into 0.5 Hz frequency bands by
    using average, thus, evaluating an average spectral power in each
    0.5 Hz frequency band from 0 to 64 Hz.
    """
    spec_1 = subsequently_binned(spec=sP, len_axis=len(freq), step=4, type_axis=0, start=1, num_elements=4)

    freq_1 = np.array(np.arange(0.0, spec_1.shape[0])) * 0.5

    return spec_1, freq_1


def get_matrix_feature(spec_1c, spec_2c):
    vectorf_part1 = 10 * (np.log10(spec_1c))

    vectorf_part2 = 10 * (np.log10(spec_2c))

    matrix_feature = np.concatenate((vectorf_part1, vectorf_part2), axis=1)

    i = 0

    return matrix_feature


def get_classes(dim_V1, dim_V2):
    vectorc_part1 = np.array(['attentive' for i in range(dim_V1)])

    vectorc_part2 = np.array(['distracted' for i in range(dim_V2)])

    vector_classes = np.concatenate((vectorc_part1, vectorc_part2))

    return vector_classes
