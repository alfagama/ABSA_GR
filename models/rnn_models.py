from models.rnn_models_list import *

import pandas as pd
import numpy as np
import os
import time

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten, Dropout, Bidirectional, GRU
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from numpy import zeros
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import Counter
import pickle
from sklearn import metrics

import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def rnn_methods():
    directory_og = '..//final_datasets/og/'
    directory_xx = '..//final_datasets/xx/'
    directory_15 = '..//final_datasets/15/'
    directories = [directory_og, directory_xx, directory_15]

    train_list = []
    test_list = []

    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.__contains__('01'):
                    pass
                elif file.__contains__('train'):
                    path = directory + file
                    train_list.append(path)
                elif file.__contains__('test'):
                    path = directory + file
                    test_list.append(path)

    train_list = sorted(train_list)
    test_list = sorted(test_list)

    for i, item in enumerate(train_list):
        train = pd.read_csv(item, sep=',')
        test = pd.read_csv(test_list[i], sep=',')
        #
        print(item)
        # text
        train_words_counter = Counter(word for sentence in list(train['text']) for word in sentence.split())
        train_words_count = len(train_words_counter)
        sentence_list = [[s for s in list.split()] for list in train['text']]  # convert to list of lists of words
        max_length = len(max(sentence_list, key=len))  # find the max length of list
        test_words_counter = Counter(word for sentence in list(test['text']) for word in sentence.split())
        # train_words_count = sorted(train_words_counter.items(), key=lambda kv: kv[1])
        test_words_counter = len(test_words_counter)
        sentence_list_test = [[s for s in list.split()] for list in test['text']]  # convert to list of lists of words
        max_length_test = len(max(sentence_list_test, key=len))  # find the max length of list
        # target
        train_target_counter = Counter(word for sentence in list(train['target']) for word in sentence.split())
        train_target_count = len(train_target_counter)
        sentence_list = [[s for s in list.split()] for list in train['target']]  # convert to list of lists of words
        max_length_target = len(max(sentence_list, key=len))  # find the max length of list
        test_target_counter = Counter(word for sentence in list(test['target']) for word in sentence.split())
        # train_words_count = sorted(train_words_counter.items(), key=lambda kv: kv[1])
        test_target_count = len(test_target_counter)
        sentence_list_test = [[s for s in list.split()] for list in test['target']]  # convert to list of lists of words
        max_length_target_test = len(max(sentence_list_test, key=len))  # find the max length of list
        #
        padded_lenth = max_length + max_length_target
        #
        tokenizer = Tokenizer(num_words=(train_words_count), split=' ', oov_token="<UKN>")
        tokenizer.fit_on_texts(train['text'])
        #
        train['conc'] = train[['text', 'target']].agg(' '.join, axis=1)
        X_train2 = tokenizer.texts_to_sequences(train['conc'])
        X_train2 = pad_sequences(X_train2, maxlen=padded_lenth)
        #
        test['conc'] = test[['text', 'target']].agg(' '.join, axis=1)
        X_test2 = tokenizer.texts_to_sequences(test['conc'])
        X_test2 = pad_sequences(X_test2, maxlen=padded_lenth)
        #
        # Y_train2 = train['sentiment'].values
        Y_train2 = pd.get_dummies(train['sentiment']).values
        #
        # Y_test2 = pd.get_dummies(test['sentiment']).values
        # print('Shape of label tensor:', Y_test2.shape)
        Y_test2 = test['sentiment'].values
        #

        #################################################################################
        #                                                                               #
        model = Sequential()                                                            #
        #                                                                               #
        #                                                                               #
        #            INSERT MODEL HERE FROM --> rnn_models_list.py                      #
        #                                                                               #
        #                                                                               #
        #################################################################################

        #
        epochs = 4
        batch_size = 64
        #
        history = model.fit(X_train2,
                            Y_train2,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1, )
        # callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        #
        model.save('..//path/model_name.h5')

        #
        # model_loaded = tf.keras.models.load_model('..//path/model_name.h5')

        #
        y_pred = model.predict(X_test2)
        #
        sent_pred = []
        for i in range(0, len(y_pred)):
            pred = np.argmax(y_pred[i])
            sent_pred.append(pred)
        #
        print(metrics.accuracy_score(Y_test2, sent_pred))
        print(metrics.precision_score(Y_test2, sent_pred, average="macro"))
        print(metrics.recall_score(Y_test2, sent_pred, average="macro"))
        print(metrics.f1_score(Y_test2, sent_pred, average="macro"))
        print(metrics.f1_score(Y_test2, sent_pred, average="micro"))
        print(metrics.f1_score(Y_test2, sent_pred, average="weighted"))
        print(metrics.classification_report(Y_test2, sent_pred))

        cf_matrix = confusion_matrix(Y_test2, sent_pred)
        sb.set(font_scale=1)
        heat_map = sb.heatmap(cf_matrix,
                              # xticklabels="Predicted",
                              # yticklabels="Golen Label",
                              cmap="YlGnBu",
                              fmt='',
                              annot=True,
                              cbar=False)
        fig = plt.gcf()

        # plt.title('Heatmap of Flighr Dataset', fontsize = 20) # title with fontsize 20
        plt.xlabel('Predicted', fontsize=15)  # x-axis label with fontsize 15
        plt.ylabel('Golden Label', fontsize=15)  # y-axis label with fontsize 15

        m_name = time.strftime("%Y%m%d-%H%M%S")
        p_name = "'..//path/figures/" + m_name + ".png'"
        plt.savefig(p_name, dpi=600)
        plt.show()
