import random
from collections import defaultdict, OrderedDict

from config import GLOBAL_MAX_LEN, N_LABELS
from time import time
import random
import tqdm

import os
import numpy as np
import pickle
import datetime
from sklearn.metrics import precision_recall_fscore_support

import cv2
import keras
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import tensorflow as tf
from third_parties.TT_RNN.TTRNN import TT_GRU, TT_LSTM
import argparse


parser = argparse.ArgumentParser(description='DeepSBD experiment argument parser.')
parser.add_argument('--save_weights_path', required=True, help='Path to savw weights during training.')
parser.add_argument('--pretrained_weights_path', default=None, type=str, help='Path to savw weights during training.')
parser.add_argument('--rnn_layer', default='lstm', choices=['lstm', 'gru'], help='Type of RNN unit. Can be either lstm or gru.')
parser.add_argument('--use_TT', action='store_constant', default=False, const=True, dest='use_TT', help='Flag to use TT-decomposition of rnn weights.')
parser.add_argument('--data_path', type=str, help='DeepSBD dataset root.')
parser.add_argument('--labels_path', type=str, help='DeepSBD train labels path.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate value.')
parser.add_argument('--dropout_rate', default=0.25, type=float, help='Dropout rate.')
parser.add_argument('--batch_size', default=300, type=int, help='Batch size.')
parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs to train.')
parser.add_argument('--n_workers', default=4, type=int, help='Number of workers.')

args = parser.parse_args()


def main(args):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    np.random.seed(11111986)

    alpha = 1e-2
    # decomposition is fixed for DeepSBD dataset
    tt_input_shape = [10, 18, 13, 30]
    tt_input_shape = [7, 16, 12, 28]
    tt_output_shape = [4, 4, 4, 4]
    tt_ranks = [1, 4, 4, 4, 1]


    input = Input(shape=(16, 112*112*3))
    if args.rnn_layer == 'gru':
        if not args.use_TT:
            rnn_layer = GRU(np.prod(tt_output_shape),
                            return_sequences=False,
                            dropout=args.dropout_rate, recurrent_dropout=args.dropout_rate, activation='tanh')
        else:
            rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                               tt_ranks=tt_ranks,
                               return_sequences=False,
                               dropout=args.dropout_rate, recurrent_dropout=args.dropout_rate, activation='tanh')
    else:
        if not args.use_TT:
            rnn_layer = LSTM(np.prod(tt_output_shape),
                             return_sequences=False,
                             dropout=args.dropout_rate, recurrent_dropout=args.dropout_rate, activation='tanh')
        else:
            rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                                tt_ranks=tt_ranks,
                                return_sequences=False,
                                dropout=args.dropout_rate, recurrent_dropout=args.dropout_rate, activation='tanh')
    h = rnn_layer(input)

    output = Dense(units=3, activation='softmax', kernel_regularizer=l2(alpha))(h)
    model = Model(input, output)
    model.compile(optimizer=Adam(args.lr), loss='binary_crossentropy')
    if args.pretrained_weights_path is not None:
        model.load_weights('args.pretrained_weights_path')

    dgen = ShotDataset(args.labels_path, args.data_path, batch_size=args.batch_size)
    for i in range(args.n_epochs):
        model.fit_generator(dgen, nb_epoch=1, verbose=1, workers=args.n_workers, use_multiprocessing=True)
        model.save(args.save_weights_path + '/tt_rnn_model%s.h5' % random.randint(1, 100000))

if __name__ == '__main__':
    main(args)