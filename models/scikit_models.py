import pandas as pd
import numpy as np
import os
import time

import fasttext as ft
import fasttext.util

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def fasttext600(train, test):
    print("Train length: ", len(train), " Test length: ", len(test))
    # prepare train (X_train / Y_train)
    textVectorized = []
    targetVectorized = []
    for sentence in train['text']:
        textVectorized.append(ft.get_sentence_vector(sentence))
    for target in train['target']:
        targetVectorized.append(ft.get_sentence_vector(target))
    df_text = pd.DataFrame(textVectorized)
    df_target = pd.DataFrame(targetVectorized)
    df_fin = pd.concat([df_text, df_target, train['sentiment']], axis=1)
    X_train = df_fin.values[:, -len(df_fin.values):-1]
    Y_train = df_fin.values[:, -1]
    # prepare test (X_test / Y_test)
    textVectorized_test = []
    targetVectorized_test = []
    for sentence in test['text']:
        textVectorized_test.append(ft.get_sentence_vector(sentence))
    for target in test['target']:
        targetVectorized_test.append(ft.get_sentence_vector(target))
    df_text_test = pd.DataFrame(textVectorized_test)
    df_target_test = pd.DataFrame(targetVectorized_test)
    df_fin_test = pd.concat([df_text_test, df_target_test, test['sentiment']], axis=1)
    X_test = df_fin_test.values[:, -len(df_fin_test.values):-1]
    Y_test = df_fin_test.values[:, -1]
    return X_train, Y_train, X_test, Y_test


def fasttext39300(train, test):
    # prepare train (X_train / Y_train)
    textVectorized = []
    targetVectorized = []
    # for sentence in train['text']:
    #   words = [word.strip() for word in train.iloc[i]['targets'].split(' ')]
    for i in range(0, len(train)):
        words = [word.strip() for word in train.iloc[i]['text'].split(' ')]
        for word in words:
            textVectorized.append(ft.get_word_vector(word))
        if len(words) < 130:
            extra_words = 130 - len(words)
            for x in range(0, extra_words):
                textVectorized.append(np.zeros(300))
    for target in train['target']:
        target_words = [word.strip() for word in train.iloc[i]['target'].split(' ')]
        for t in target_words:
            targetVectorized.append(ft.get_word_vector(word))
        if len(target_words) < 130:
            extra_words_t = 130 - len(target_words)
            for x in range(0, extra_words_t):
                targetVectorized.append(np.zeros(300))
    df_text = pd.DataFrame(textVectorized)
    df_target = pd.DataFrame(targetVectorized)
    df_fin = pd.concat([df_text, df_target, train['sentiment']], axis=1)
    X_train = df_fin.values[:, -len(df_fin.values):-1]
    Y_train = df_fin.values[:, -1]
    # prepare test (X_test / Y_test)
    textVectorized_test = []
    targetVectorized_test = []
    for i in range(0, len(test)):
        words = [word.strip() for word in test.iloc[i]['text'].split(' ')]
        for word in words:
            textVectorized_test.append(ft.get_word_vector(word))
        if len(words) < 130:
            extra_words = 130 - len(words)
            for x in range(0, extra_words):
                textVectorized_test.append(np.zeros(300))
    for target in test['target']:
        target_words = [word.strip() for word in test.iloc[i]['target'].split(' ')]
        for t in target_words:
            targetVectorized_test.append(ft.get_word_vector(word))
        if len(target_words) < 130:
            extra_words_t = 130 - len(target_words)
            for x in range(0, extra_words_t):
                targetVectorized_test.append(np.zeros(300))
    df_text_test = pd.DataFrame(textVectorized_test)
    df_target_test = pd.DataFrame(targetVectorized_test)
    df_fin_test = pd.concat([df_text_test, df_target_test, test['sentiment']], axis=1)
    X_test = df_fin_test.values[:, -len(df_fin_test.values):-1]
    Y_test = df_fin_test.values[:, -1]
    return X_train, Y_train, X_test, Y_test


def scikit_methods():
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
        X_train, Y_train, X_test, Y_test = fasttext600(train, test)
        # X_train, Y_train, X_test, Y_test = fasttext39300(train, test)
        # Show dataset name
        print(item)
        # Run SVM
        print('Running SVM ...')
        start = time.time()
        parameters = {
            'C': [100],
            'kernel': ['rbf'],
            'degree': [2],
            'gamma': [3],
            'random_state': [11]
        }
        model = svm.SVC()
        svc_model = GridSearchCV(model, parameters)
        svc_model.fit(X_train, Y_train)
        y_predicted = svc_model.predict(X_test)
        print(len(y_predicted))
        print(metrics.accuracy_score(Y_test, y_predicted))
        print(metrics.precision_score(Y_test, y_predicted, average="macro"))
        print(metrics.recall_score(Y_test, y_predicted, average="macro"))
        print(metrics.f1_score(Y_test, y_predicted, average="macro"))
        print(metrics.f1_score(Y_test, y_predicted, average="micro"))
        print(metrics.f1_score(Y_test, y_predicted, average="weighted"))
        print(metrics.classification_report(Y_test, y_predicted))
        end = round(time.time() - start, 2)
        print("This process took", end, "seconds.")
        # Run MLP
        print('Running MLP ...')
        start = time.time()
        parameters_mlp = {
            'hidden_layer_sizes': [[100]],  # [[100, 100, 100]],
            'activation': ['relu'],
            'solver': ['adam'],
            'tol': [0.0001],  # [0.00001],
            'max_iter': [100],  # [1000],
            'random_state': [11]
        }
        model = MLPClassifier()
        mlp_model = GridSearchCV(model, parameters_mlp)
        mlp_model.fit(X_train, Y_train)
        y_predicted = mlp_model.predict(X_test)
        print(metrics.accuracy_score(Y_test, y_predicted))
        print(metrics.precision_score(Y_test, y_predicted, average="macro"))
        print(metrics.recall_score(Y_test, y_predicted, average="macro"))
        print(metrics.f1_score(Y_test, y_predicted, average="macro"))
        print(metrics.f1_score(Y_test, y_predicted, average="micro"))
        print(metrics.f1_score(Y_test, y_predicted, average="weighted"))
        print(metrics.classification_report(Y_test, y_predicted))
        end = round(time.time() - start, 2)
        print("This process took", end, "seconds.")
