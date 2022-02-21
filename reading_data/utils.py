import pandas as pd
import time
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split


def save(df):
    dir = 'data_custom/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    name = time.strftime("%Y%m%d-%H%M%S")
    path = dir + name + '.csv'
    df.to_csv(path)


def save_phase2(df):
    directory = 'data_phase2/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = time.strftime("%Y%m%d-%H%M%S")
    path = directory + 'phase2-' + name + '.csv'
    df.to_csv(path)


def save_test(df, name):
    directory = 'data_phase3/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + name + '.csv'
    df.to_csv(path)


def save_imbalance(df):
    directory = 'data_custom_imbalanced/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = time.strftime("%Y%m%d-%H%M%S")
    path = directory + 'cust_imb-' + name + '.csv'
    df.to_csv(path)


def reset_indx(df):
    df = df.reset_index(drop=True)
    return df


def remove_empty_rows(df, col):
    df.dropna(subset=[col], inplace=True)
    return df


def lower_case_col(data, col, new_col):
    data[new_col] = data[col].str.lower()
    return data


def replace_minus_minus_two(data):
    data['sentiment'] = data['sentiment'].replace('--2', '-2')
    return data


def print_zip(data):
    unique, counts = np.unique(data['sentiment'], return_counts=True)
    print(dict(zip(unique, counts)))


def check_if_cols_are_identical(df1, df2, check, show_errors):
    for col in df1:
        print(col, ' ', df1[col].equals(df2[col]))
    if not df1['text'].equals(df2['text']) and check:
        if show_errors:
            for i in range(0, len(df1)):
                if df1['text'].iloc[i] != df2['text'].iloc[i]:
                    print("Error in line: ", i)
                    print(df1['text'].iloc[i])
                    print(df2['text'].iloc[i])
                    print(df1['targets'].iloc[i])
                    print(df2['targets'].iloc[i])
        GK_df = df1.drop(df1[df1.text != df2.text].index)
        GR_df = df2.drop(df2[df2.text != df1.text].index)

        return GK_df, GR_df
    return df1, df2


def check_if_targets_are_NA(df1, df2):
    df1 = df1[df1['targets'].notna()]
    df2 = df2[df2['targets'].notna()]
    return df1, df2


def keep_same_ids(df1, df2):
    data = pd.merge(df1, df2, on='id', how='inner')
    return data


def drop_na_sentiments(new_df):
    new_df_dropNA = new_df[new_df['sentiments'] != '']
    return new_df_dropNA


def discard_mismatch_in_sentiments_and_targets(GK_df_notNA, GR_df_notNA):
    # Drop len(sentiment) != len(target)
    GK_ids_to_drop = []
    GK_all_ids = []
    for i in range(len(GK_df_notNA)):
        GK_all_ids.append(GK_df_notNA.iloc[i]['id'])
        if str(GK_df_notNA['targets'].iloc[i]).count(',') != str(GK_df_notNA['sentiments'].iloc[i]).count(','):
            # print("Error in line: ", i, GK_df_notNA['text'].iloc[i])
            # print(GK_df_notNA['targets'].iloc[i])
            # print(GK_df_notNA['sentiments'].iloc[i])
            GK_ids_to_drop.append(GK_df_notNA.iloc[i]['id'])
    GK_df_notNA_notST = GK_df_notNA[~GK_df_notNA['id'].isin(list(GK_ids_to_drop))]

    GR_ids_to_drop = []
    GR_all_ids = []
    for i in range(len(GR_df_notNA)):
        GR_all_ids.append(GR_df_notNA.iloc[i]['id'])
        if str(GR_df_notNA['targets'].iloc[i]).count(',') != str(GR_df_notNA['sentiments'].iloc[i]).count(','):
            # print("Error in line: ", i, GR_df_notNA['text'].iloc[i])
            # print(GR_df_notNA['targets'].iloc[i])
            # print(GR_df_notNA['sentiments'].iloc[i])
            GR_ids_to_drop.append(GR_df_notNA.iloc[i]['id'])
    GR_df_notNA_notST = GR_df_notNA[~GR_df_notNA['id'].isin(list(GR_ids_to_drop))]

    return GK_df_notNA_notST, GR_df_notNA_notST


def check_if_target_is_included_in_text(data, target, text):
    ids_that_do_not_match_targetWord_in_sentence = []
    # for row in range(0, len(df['preText_NA'])):
    for ind, row in enumerate(data[target]):
        if data.iloc[ind][target] not in data.iloc[ind][text]:
            # print(df.iloc[row]['target_word_LC_NA'])
            # print(data.iloc[row]['target_word_og'])
            print(data.iloc[ind]['id'])
            # print(df.iloc[row]['sentiment'])
            ids_that_do_not_match_targetWord_in_sentence.append(ind)
    print(len(ids_that_do_not_match_targetWord_in_sentence))


def count_distinct_word(data, col_name):
    train_words_counter = Counter(word for sentence in list(data[col_name]) for word in sentence.split())
    train_words_count = len(train_words_counter)
    print("Total number of distinct words in col: ", col_name, ' is: ', train_words_count)


def max_length(data, col_name):
    sentence_list = [[s for s in l.split()] for l in data[col_name]]
    sent_max_length = len(max(sentence_list, key=len))
    print("Max length: ", sent_max_length)


def split(X, y, SEED, percentage):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=percentage, random_state=SEED)
    return X_train, X_test, Y_train, Y_test


def split_df(X, SEED, percentage):
    train, test = train_test_split(X, test_size=percentage, random_state=SEED)
    return train, test


def split_to_X_and_y(data):
    X = data.values[:, -len(data.values):-1]
    y = data.values[:, -1]
    return X, y

def replace_sentiment_range(df):
    key_value_dict = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    changer = {v: k for k, v in key_value_dict.items()}
    df['sentiment'] = df['sentiment'].replace(changer)
    return df