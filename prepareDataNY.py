from utils import *


def load_data_NY(filename, T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, meta_data=True):
    assert (len_closeness + len_period + len_trend > 0)
    dir = os.getcwd()
    print('print dir:', dir)
    filepath = os.path.join(dir, 'data','BikeNYC',filename)
    # load data
    data, timestamps = load_stdata(filepath)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []

    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]

    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


T = 24  # number of time intervals in one day
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 10 # divide data into two subsets: Train & Test, of which the test set is the last 10 days
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
original_filename = 'NYC14_M16x8_T60_NewEnd.h5'
consider_external_info = True

if consider_external_info:
    # generate data with external information
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data_NY(original_filename, T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
                     len_trend=len_trend, len_test=len_test, meta_data=True)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'BikeNYC', 'NYC_c%d_p%d_t%d_ext' % (len_closeness, len_period, len_trend))
    print('filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()

else:
    # generate data without external information
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
            load_data_NY(original_filename, T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
                      len_trend=len_trend, len_test=len_test, meta_data=False)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'BikeNYC','NYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
    print('filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()


