import fasttext
import numpy as np
import pandas as pd


def textToVector_perWord(row):
    word_vec_list = []
    words = row.split()
    for word in words:
        word_vec = ft.get_word_vector(word)
        word_vec_list.append(word_vec)
    return word_vec_list


def textToVector_perWord_method(data, old_col, new_col):
    data[new_col] = data.apply(lambda row: textToVector_perWord(row[old_col]), axis=1)
    return data


def textToVectorPadded_perWord(row, max_length, pad_length):
    padded_list = []
    words = row.split()
    for word in words:
        word_vec = ft.get_word_vector(word)
        padded_list.append(word_vec)
    if len(words) < max_length:
        for i in range(0, (max_length - len(words))):
            padded_list.append(np.zeros([pad_length]))
    return padded_list


def textToVectorPadded_perWord_method(data, old_col, new_col, max_length, pad_length):
    data[new_col] = data.apply(lambda row: textToVectorPadded_perWord(row[old_col], max_length, pad_length), axis=1)
    return data


def sentenceVectorized(data, text_col, target_col):
    textVectorized = []
    targetVectorized = []
    for sentence in data[text_col]:
        textVectorized.append(ft.get_sentence_vector(sentence))
    for target in data[target_col]:
        targetVectorized.append(ft.get_sentence_vector(target))

    df_text = pd.DataFrame(textVectorized)
    df_target = pd.DataFrame(targetVectorized)

    df_fin = pd.concat([df_text, df_target, data['sentiment']], axis=1)

    return df_fin


def wordbywordVectorized_method(row, max_length):
    padded_list = []
    words = row.split()
    for word in words:
        word_vec = ft.get_word_vector(word)
        padded_list.append(word_vec)
    if len(words) < max_length:
        for i in range(0, (max_length - len(words))):
            padded_list.append(np.zeros([300]))
    return padded_list


def wordbywordVectorized(data, new_name, col, max_length):
    data[new_name] = data.apply(lambda row: wordbywordVectorized_method(row[col], max_length), axis=1)
    return data
