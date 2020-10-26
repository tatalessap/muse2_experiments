import matplotlib.pyplot as plt
from utils import *


# /home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/
def generate_by_all_files(path_file_csv, name_file, smooth=False):
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
        complete_matrix, vector_classes = get_complete_matrix(listCh, df, smooth)

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

    if smooth:
        name_file = name_file + '_with_smooth'

    big_matrix.to_csv(name_file + '.csv', index=False)  # save the file

    np.savez('index_' + name_file, np.array(indexes))


# /home/tatalessap/PycharmProjects/muse2_tesi/res_eti_256row/
def generate_by_one_file(path_file_csv, name_file, path_save, smooth=False):
    bb = []
    # the size of complete_matrix changes
    files = os.listdir(path_file_csv)
    dataSets = [pd.read_csv(path_file_csv + str(el)) for el in files]

    for df in dataSets:
        df = df[['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'label']]

        listCh = list(df.columns[:-1])

        # if ica: df = filter_with_ica(df)

        complete_matrix, vector_classes = get_complete_matrix(listCh, df, smooth)

        big_matrix = pd.DataFrame(np.transpose(complete_matrix))

        big_matrix['classes'] = list(vector_classes)

        bb.append(big_matrix)

    index_file = 0
    for big_matrix in bb:
        if smooth and index_file == 0:
            name_file = name_file + 'with_smooth'
        big_matrix.to_csv(path_save + '/' + name_file + str(index_file) + '.csv', index=False)  # save the file
        index_file = index_file + 1


"""
the complete_matrix is the matrix where is the merge of all feature_matrix. In this function there is the calculate of 
spectrogram of each task
"""


def get_complete_matrix(list_ch, data, smooth):
    last_index = 0

    complete_matrix = np.array([])

    vector_classes = np.array([])

    dfs = [x for _, x in data.groupby('label')]

    data_attentive = dfs[0]
    data_distracted = dfs[1]

    # data_attentive.to_csv('attentive' + '.csv', index=False)  # save the file
    # data_distracted.to_csv('distracted' + '.csv', index=False)  # save the file

    for el in list_ch:
        x = np.array(data_attentive[el])

        cmap = plt.get_cmap('viridis')

        cmap.set_under(color='k', alpha=None)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10))

        sP, freq, t, im = ax1.specgram(x, Fs=256, window=np.blackman(M=256), NFFT=256, pad_to=2048, noverlap=0)

        ax1.set(title='attentive')

        spec_a, freq_ = steps(sP, freq, smooth)

        # second

        x = np.array(data_distracted[el])

        sP, freq, t, im = ax2.specgram(x, Fs=256, window=np.blackman(M=256), NFFT=256, pad_to=2048, noverlap=0)

        spec_d, freq_ = steps(sP, freq, smooth)

        ax2.set(title='distracted')

        feature_matrix = get_matrix_feature(spec_a, spec_d)

        # plt.close(fig)

        fig.colorbar(im, ax=ax1)
        fig.colorbar(im, ax=ax2)

        plt.close(fig)

        # plt.show()

        vector_classes = get_classes(spec_a.shape[1], spec_d.shape[1])

        # merge of feature_matrix of the single channel
        complete_matrix, last_index = add_matrix_to_matrix(complete_matrix, feature_matrix, len(list_ch), last_index)

    return complete_matrix, vector_classes


def steps(sP, freq, smooth):
    """
    These were subsequently binned into 0.5 Hz frequency bands by
    using average, thus, evaluating an average spectral power in each
    0.5 Hz frequency band from 0 to 64 Hz.
    """
    spec_1 = subsequently_binned(spec=sP, len_axis=len(freq), step=4, type_axis=0, start=1, num_elements=4)

    freq_1 = np.array(np.arange(1, spec_1.shape[0] + 1)) * 0.5

    # step2
    """
    The frequency range was
    then restricted to 0–18 Hz so that only 36 frequencies,  k = k · 0.5
    Hz, k = 1,...,16, were retained in the dataset. The constant compo-
    nent  = 0 Hz was discarded
    """
    result = np.where(freq_1 == 18)

    spec_2 = spec_1[0:int(result[0]) + 1, :]

    freq_2 = freq_1[0:int(result[0]) + 1]

    # step3
    """
    Finally, the binned and frequency-
    restricted spectrograms S ( t ,  ) were temporally smoothed by using
    a 15 s-running average.
    to fix
    """
    if smooth:
        spec_second3 = np.transpose(
            subsequently_binned(spec=np.transpose(spec_2), len_axis=len(spec_2[1]), step=1, type_axis=0, start=1,
                                num_elements=15))

        spec = np.insert(spec_second3, 0, values=np.transpose(spec_2[:, [0]]), axis=1)
    else:
        spec = spec_2

    return spec, freq_2


def get_matrix_feature(spec_1c, spec_2c):
    vectorf_part1 = 10 * (np.log10(spec_1c))

    vectorf_part2 = 10 * (np.log10(spec_2c))

    matrix_feature = np.concatenate((vectorf_part1, vectorf_part2), axis=1)

    return matrix_feature


def get_classes(dim_V1, dim_V2):
    vectorc_part1 = np.array(['attentive' for i in range(dim_V1)])

    vectorc_part2 = np.array(['distracted' for i in range(dim_V2)])

    vector_classes = np.concatenate((vectorc_part1, vectorc_part2))

    return vector_classes
