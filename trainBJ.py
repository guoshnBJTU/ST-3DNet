from ST3DNet import *
from utils import *
import argparse
import configparser
import os
import time
from keras.utils import plot_model
from keras.optimizers import Adam

dir = os.getcwd()

# parser param
parser = argparse.ArgumentParser()
parser.add_argument("--ctx", default='1', type=str)
args = parser.parse_args()
ctx = args.ctx
os.environ["CUDA_VISIBLE_DEVICES"] = ctx

# read config file
config_file = os.path.join(dir, 'config', 'bj.conf')
config = configparser.ConfigParser()
print('Read configuration file: %s' % config_file)
print('>>>>>>>  configuration   <<<<<<<')
with open(config_file, 'r') as f:
    print(f.read())
print('\n')
config.read(config_file)
training_config = config['Training']

lr = float(training_config['learning_rate'])
batch_size = int(training_config['batch_size'])
nb_residual_unit = int(training_config['nb_residual_unit'])  # number of residual units

nb_epoch = int(training_config['nb_epoch'])  # number of epoch at training stage
consider_external_info = bool(int(training_config['consider_external_info']))

len_closeness = int(training_config['len_closeness'])  # length of closeness dependent sequence
len_period = int(training_config['len_period'])  # length of peroid dependent sequence
len_trend = int(training_config['len_trend'])  # length of trend dependent sequence

cs = [True, False]
for p in cs:
    consider_external_info = p
    filename = 'TaxiBJ_c%d_p%d_t%d' % (len_closeness, len_period, len_trend)
    hyperparams_name = 'TaxiBJ_c%d_p%d_t%d_r%d_b%d_lr%.1e' % (len_closeness, len_period, len_trend, nb_residual_unit, batch_size, lr)
    T = 48  # number of time intervals in one day
    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 7 * 4  # divide data into two subsets: Train & Test, of which the test set is the last 4 weeks
    len_test = T * days_test
    map_height, map_width = 32, 32  # grid size

    if consider_external_info:
        filename = filename + '_ext'
        hyperparams_name = hyperparams_name + '_ext'
    else:
        filename = filename + '_noext'
        hyperparams_name = hyperparams_name + '_noext'

    filename = os.path.join(dir, "data", 'TaxiBJ', filename)
    expdir = os.path.join(dir, "experiment", 'TaxiBJ')
    fname_param = os.path.join(expdir, hyperparams_name + '.best.h5')
    print('filename:', filename)
    print('fname_param:', fname_param)

    f = open(filename, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    mmn = pickle.load(f)
    external_dim = pickle.load(f)
    timestamp_train = pickle.load(f)
    timestamp_test = pickle.load(f)

    for i in X_train:
        print(i.shape)

    Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
    Y_test = mmn.inverse_transform(Y_test)

    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None

    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)

    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])
    model.summary()
    plot_model(model, to_file=os.path.join(expdir,'BJmodel.png'), show_shapes=True)

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    fname_param = os.path.join(expdir, hyperparams_name + '.best.h5')

    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=1, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    start_time = time.time()
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[model_checkpoint],
                        verbose=2)

    end_time = time.time()
    print('cost %.f mins on training'% ((end_time-start_time)//60))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f  rmse (real): %.6f' %(score[0], score[1]))
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f  rmse (real): %.6f' %(score[0], score[1]))

    Y_test_predict = model.predict(X_test, batch_size=Y_test.shape[0], verbose=0)
    test_rmse, test_mae, test_mape = compute(Y_test, Y_test_predict)
    print(hyperparams_name+'rmse:%.6f, mae:%.6f, mape:%.6f' % (test_rmse, test_mae, test_mape))

    print('=' * 10)
    print("cont training model...")
    start_time = time.time()
    adam = Adam(lr=0.1 * lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])
    model.load_weights(fname_param)
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=1, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train,
                        nb_epoch=60,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint],
                        verbose=2)

    end_time = time.time()
    print('cost %.f mins on training' % ((end_time - start_time) // 60))

    print('=' * 10)
    print('cont evaluating ...')
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('cont Train score: %.6f  rmse (real): %.6f' % (score[0], score[1]))
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('cont Test score: %.6f  rmse (real): %.6f' % (score[0], score[1]))

    Y_test_predict = model.predict(X_test, batch_size=Y_test.shape[0], verbose=0)
    test_rmse, test_mae, test_mape = compute(Y_test, Y_test_predict)
    print(hyperparams_name + 'rmse:%.6f, mae:%.6f, mape:%.6f' % (test_rmse, test_mae, test_mape))

