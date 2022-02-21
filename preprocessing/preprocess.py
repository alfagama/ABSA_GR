import pandas as pd
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import nltk
import regex as re
import unicodedata
from preprocessing.nlp_preprocessing import *


def clear_target_col(df_pre):
    df_pre['targets_GK_og'] = df_pre['targets_GK']
    df_pre['targets_GR_og'] = df_pre['targets_GR']
    df_pre['targets_GK'] = df_pre['targets_GK'].str.replace('#', '')
    df_pre['targets_GK'] = df_pre['targets_GK'].str.replace('@', '')
    df_pre['targets_GR'] = df_pre['targets_GR'].str.replace('#', '')
    df_pre['targets_GR'] = df_pre['targets_GR'].str.replace('@', '')
    df_pre['targets_GK'] = df_pre['targets_GK'].str.replace(', ', ',')
    df_pre['targets_GK'] = df_pre['targets_GK'].str.replace(' ,', ',')
    df_pre['targets_GR'] = df_pre['targets_GR'].str.replace(', ', ',')
    df_pre['targets_GR'] = df_pre['targets_GR'].str.replace(' ,', ',')

    df_pre['targets_GK'] = df_pre['targets_GK'].str.lower()
    df_pre['targets_GR'] = df_pre['targets_GR'].str.lower()

    df_pre['sentiments_GK'] = df_pre['sentiments_GK'].astype('string')
    df_pre['sentiments_GR'] = df_pre['sentiments_GR'].astype('string')

    df_pre['sentiments_GK'] = df_pre['sentiments_GK'].str.replace(', ', ',')
    df_pre['sentiments_GR'] = df_pre['sentiments_GR'].str.replace(', ', ',')
    df_pre['sentiments_GK'] = df_pre['sentiments_GK'].str.replace(' ,', ',')
    df_pre['sentiments_GR'] = df_pre['sentiments_GR'].str.replace(' ,', ',')

    return df_pre


def keep_only_agreed_upon_sentiments(df_pre):
    column_names = ["id", "createdate", "channel", "text", "target_words", "sentiments", "targets_og"]
    new_df = pd.DataFrame(columns=column_names)

    # targets_GK_new = []
    # sentiments_GK_new = []
    # targets_GR_new = []
    # sentiments_GR_new = []

    targets_row_list = []
    sentiments_row_list = []
    target_og_row_list = []

    # df_pre.iloc[2875]['targets_GK'] = 'ena,dyo,tria,tessera,eksi'
    # df_pre.iloc[2875]['targets_GR'] = 'dyo,tessera, pente'
    # df_pre.iloc[2875]['sentiments_GK'] = '1,0,-2,1,1'
    # df_pre.iloc[2875]['sentiments_GR'] = '0,1,2'

    df_pre['sentiments_GK'] = df_pre['sentiments_GK'].astype(str)
    df_pre['sentiments_GR'] = df_pre['sentiments_GR'].astype(str)
    df_pre['targets_GK'] = df_pre['targets_GK'].astype(str)
    df_pre['targets_GR'] = df_pre['targets_GR'].astype(str)

    for i in range(0, len(df_pre)):
        if df_pre.iloc[i]['targets_GK'] != df_pre.iloc[i]['targets_GR']:
            # i = 2875
            targets_GK = [word.strip() for word in df_pre.iloc[i]['targets_GK'].split(',')]
            targets_GR = [word.strip() for word in df_pre.iloc[i]['targets_GR'].split(',')]
            targets_GK_og = [word.strip() for word in df_pre.iloc[i]['targets_GK_og'].split(',')]
            # targets_GR_og = [word.strip() for word in df_pre.iloc[i]['targets_GR_og'].split(',')]
            sentiments_GK = [word.strip() for word in df_pre.iloc[i]['sentiments_GK'].split(',')]
            sentiments_GR = [word.strip() for word in df_pre.iloc[i]['sentiments_GR'].split(',')]

            for place, target in enumerate(targets_GK):
                if target in targets_GR:
                    # if
                    GR_sent_place = targets_GR.index(target)
                    if sentiments_GK[place] == sentiments_GR[GR_sent_place]:
                        # print(target, sentiments_GK[place], sentiments_GR[GR_sent_place])

                        targets_row_list.append(target)
                        sentiments_row_list.append(sentiments_GK[place])
                        target_og_row_list.append(targets_GK_og[place])

            targets_new = ','.join(targets_row_list)
            targets_og_new = ','.join(target_og_row_list)
            sentiments_new = ','.join(sentiments_row_list)

            new_df = new_df.append({'id': df_pre.iloc[i]['id'],
                                    'createdate': df_pre.iloc[i]['createdate'],
                                    'channel': df_pre.iloc[i]['channel'],
                                    'text': df_pre.iloc[i]['text'],
                                    'target_words': targets_new,
                                    'sentiments': sentiments_new,
                                    'targets_og': targets_og_new
                                    },
                                   ignore_index=True)

            targets_row_list.clear()
            sentiments_row_list.clear()
            target_og_row_list.clear()

        elif df_pre.iloc[i]['sentiments_GK'] != df_pre.iloc[i]['sentiments_GR']:
            pass

        elif df_pre.iloc[i]['sentiments_GK'] == df_pre.iloc[i]['sentiments_GR']:
            new_df = new_df.append({'id': df_pre.iloc[i]['id'],
                                    'createdate': df_pre.iloc[i]['createdate'],
                                    'channel': df_pre.iloc[i]['channel'],
                                    'text': df_pre.iloc[i]['text'],
                                    'target_words': df_pre.iloc[i]['targets_GK'],
                                    'sentiments': df_pre.iloc[i]['sentiments_GK'],
                                    'targets_og': df_pre.iloc[i]['targets_GK_og']
                                    },
                                   ignore_index=True)

    # new_df = new_df.drop(columns=['createdate_x', 'sentiment_GK', 'sentiment_GR'])
    return new_df


def unroll_sentiments(new_df):
    column_names = ["id", "createdate", "channel", "text", "target_word", "target_word_og", "sentiment"]
    df_oneT_perR = pd.DataFrame(columns=column_names)

    for i in range(0, len(new_df)):
        row_targets = [word.strip() for word in new_df.iloc[i]['target_words'].split(',')]
        row_targets_og = [word.strip() for word in str(new_df.iloc[i]['targets_og']).split(',')]
        row_sentiments = [word.strip() for word in new_df.iloc[i]['sentiments'].split(',')]

        for x, targeted_word in enumerate(row_targets):
            try:
                df_oneT_perR = df_oneT_perR.append({'id': new_df.iloc[i]['id'],
                                                    'createdate': new_df.iloc[i]['createdate'],
                                                    'channel': new_df.iloc[i]['channel'],
                                                    'text': new_df.iloc[i]['text'],
                                                    'target_word': targeted_word,
                                                    "target_word_og": row_targets_og[x],
                                                    'sentiment': row_sentiments[x],
                                                    },
                                                   ignore_index=True)
            except Exception as e:
                print(new_df.iloc[i]['id'])

    return df_oneT_perR


def oneline_method(row):
    new_row = re.sub(r'(\n+)', ' ', row)
    return new_row


def oneline(data, col, new_col):
    data[new_col] = data.apply(lambda row: oneline_method(row[col]), axis=1)
    return data


def one_line_pre(data):
    data['target_word_og'] = data['target_word_og'].astype(str)
    data['OG'] = 'Y'
    df = data.filter(['id', 'OG', 'text', 'target_word_og', 'sentiment'], axis=1)
    df = oneline(df, 'text', 'text_oneline')
    return df


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def remoce_accent_method(data, accent_col, new_accent):
    data[new_accent] = data.apply(lambda row: strip_accents(row[accent_col]), axis=1)
    return data


def create_new_columns_for_variations(df):
    df['text_lowered'] = df['text_oneline'].str.lower()
    df['target_lowered'] = df['target_word_og'].str.lower()
    df = remoce_accent_method(df, 'text_lowered', 'text_noaccent')
    df = remoce_accent_method(df, 'target_lowered', 'target_noaccent')
    df = replace(df, 'text_oneline', 'target_word_og', 'text_masked')
    df = adding_spaces(df, 'text_oneline', 'text_withspaces')
    df = clear_text_whole_method(df, 'text_oneline', 'text_removedspecialchars')
    df = remoce_accent_method(df, 'text_removedspecialchars', 'text_noaccentnospecials')
    return df