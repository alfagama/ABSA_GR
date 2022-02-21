from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np
import pandas as pd
import fasttext as ft
import fasttext.util
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from reading_data.utils import *


def imb_smote(X, y):
    sm = SMOTE(random_state=11)
    X_res, y_res = sm.fit_resample(X, y)
    unique_smote, counts_smote = np.unique(y_res, return_counts=True)
    print('Instances in each class after SMOTE: ', dict(zip(unique_smote, counts_smote)))
    return X_res, y_res


def imb_nearmiss(X, y, version):
    nm = NearMiss(version=version, n_neighbors=1)
    X_nm, y_nm = nm.fit_resample(X, y)
    unique_nearmiss, counts_nearmiss = np.unique(y_nm, return_counts=True)
    print('Instances in each class after Near Miss: ', dict(zip(unique_nearmiss, counts_nearmiss)))
    return X_nm, y_nm


def create_new_sentence_old(input_text, target_word):
    random_array = []
    tokens = []

    input_split = input_text.split()

    for word in input_split:
        coin_flip = random.choice([0, 1])
        if len(word) > 3 and coin_flip == 1 and word not in target_word.split():
            array_of_neighbours = ft.get_nearest_neighbors(word)
            for neighbour in array_of_neighbours:
                if neighbour[0] > 0.7:
                    random_array.append(neighbour[1])
                    break
            if not random_array:
                tokens.append(word)
            else:
                tokens.append(random.choice(random_array))
            random_array.clear()
        else:
            tokens.append(word)

    new_text = TreebankWordDetokenizer().detokenize(tokens)
    return new_text


def customImbalance(data):
    pass
    dataframes = [v for k, v in data.groupby('sentiment')]

    max_sentiment_rows = 0
    for dataframe in dataframes:
        if len(dataframe) > max_sentiment_rows:
            max_sentiment_rows = len(dataframe)

    column_names = ["id", "createdate", "channel", "text", "target_word", "sentiment", "preText", "preText_NA",
                    "target_word_LC", "target_word_LC_NA"]
    df_withNewSentences = pd.DataFrame(columns=column_names)

    counter = 0
    dataframe_num = 0
    for dataframe in dataframes:
        if len(dataframe) < max_sentiment_rows:
            new_sentences = round(max_sentiment_rows / len(dataframe))
            for count, row in enumerate(dataframe['preText_NA']):
                id = dataframe.iloc[count]['id']
                createdate = dataframe.iloc[count]['createdate']
                channel = dataframe.iloc[count]['channel']
                text = dataframe.iloc[count]['text']
                target_word = dataframe.iloc[count]['target_word']
                sentiment = dataframe.iloc[count]['sentiment']
                preText = dataframe.iloc[count]['preText']
                # preText_NA = dataframe.iloc[count]['preText_NA']
                target_word_LC = dataframe.iloc[count]['target_word_LC']
                target_word_LC_NA = dataframe.iloc[count]['target_word_LC_NA']
                for i in range(0, new_sentences - 1):
                    counter += 1
                    print('Sentences created: ', counter)
                    preText_NA = create_new_sentence_old(dataframe.iloc[count]['preText_NA'], target_word_LC_NA)
                    df_withNewSentences = df_withNewSentences.append({
                        'id': id,
                        'createdate': createdate,
                        'channel': channel,
                        'text': text,
                        'target_word': target_word,
                        'sentiment': sentiment,
                        'preText': preText,
                        'preText_NA': preText_NA,
                        'target_word_LC': target_word_LC,
                        'target_word_LC_NA': target_word_LC_NA
                    }, ignore_index=True)
            dataframe_num += 1
            save_imbalance(df_withNewSentences)
            # name = str(dataframe_num) + '.csv'
            # df_withNewSentences.to_csv(r'..//content/gdrive/MyDrive/' + name)

    return df_withNewSentences


def call_imbalance(data, imb_method):

    if imb_method == 'smote':
        X = data.values[:, -len(data.values):-1]
        y = data.values[:, -1]
        X, y = imb_smote(X, y)
    elif imb_method == 'near-miss':
        X = data.values[:, -len(data.values):-1]
        y = data.values[:, -1]
        imb_nearmiss(X, y, 3)  # 1, 2, 3
    elif imb_method == 'custom':
        custom_data = customImbalance(data)
        X = custom_data.values[:, -len(custom_data.values):-1]
        y = custom_data.values[:, -1]
    else:
        raise Exception("Wrong imbalance input!")

    return X, y


def create_new_sentence(input_text, target_word, threshold):
    random_array = []
    tokens = []

    input_split = input_text.split()
    rnd = random.uniform(0, 1)

    for word in input_split:
        if len(word) > 3 and threshold < rnd and word != target_word:
            array_of_neighbours = ft.get_nearest_neighbors(word)
            for neighbour in array_of_neighbours:
                if neighbour[0] > 0.7:
                    random_array.append(neighbour[1])
                    break
            if not random_array:
                tokens.append(word)
            else:
                tokens.append(random.choice(random_array))
            random_array.clear()
        else:
            tokens.append(word)

    new_text = TreebankWordDetokenizer().detokenize(tokens)
    return new_text


def imb_method(data):
    num_of_new_senteces = 0
    threshold = 0
    df_withNewSentences = data.copy(deep=False)
    for count, row in enumerate(data['text_oneline']):
        if data.iloc[count]['sentiment'] == 2:
            num_of_new_senteces = 3
            threshold = 0.5
        elif data.iloc[count]['sentiment'] == -2:
            num_of_new_senteces = 2
            threshold = 0.5
        elif data.iloc[count]['sentiment'] == 1:
            num_of_new_senteces = 1
            threshold = 0.5
        for i in range(0, num_of_new_senteces):
            id = data.iloc[count]['id']
            og = 'N'
            text = data.iloc[count]['text']
            target_word_og = data.iloc[count]['target_word_og']
            sentiment = data.iloc[count]['sentiment']
            input_text = data.iloc[count]['text_oneline']
            text_oneline = create_new_sentence(input_text, target_word_og, threshold)
            df_withNewSentences = df_withNewSentences.append({
                'id': id,
                'og': og,
                'text': text,
                'target_word_og': target_word_og,
                'sentiment': sentiment,
                'text_oneline': text_oneline
            }, ignore_index=True)
