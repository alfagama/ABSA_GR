import pandas as pd
import numpy as np
import regex as re
import stopwordsiso
import unicodedata
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words_el = stopwordsiso.stopwords(["el"])
ok_stop_words_el = ['δεν', 'μη', 'κακώς', 'κακως', 'καλώς', 'καλως']
stop_words_en = set(stopwords.words('english'))
ok_stop_words_en = ['not', 'but', 'won']

ok_stop_words = [*ok_stop_words_el, *ok_stop_words_en]

stop_words = [*stop_words_el, *stop_words_en]

# emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F"  # emoticons
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)


# def clear_text_whole(row_raw):
#     row_raw = re.sub(r'RT+ @\s*\S+', ' ', row_raw)
#     row_raw = re.sub(r'rt+ @\s*\S+', ' ', row_raw)
#
#     row_lowered = row_raw.lower()
#     row_split = row_lowered.split()
#     tokens = [token for token in row_split
#               if not token.startswith('http')
#               # if not token.startswith('#')
#               # if not token.startswith('@')
#               # if not token.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
#               # if token.islower()
#               ]
#     # # remove stopwords
#     # final_stop_words = [x for x in stop_words if x not in ok_stop_words]
#     # tokens_no_stopwords = [w for w in tokens if w not in final_stop_words]
#     # #   We deTokenize here in order to use RE more efficinetly
#     row = TreebankWordDetokenizer().detokenize(tokens)
#     # row = re.sub(r'[0-9]', ' ', row)
#     row = re.sub('\n', ' ', row)
#     row = re.sub(r'(\n+)', ' ', row)
#     row = re.sub(r'[!]', ' ! ', row)
#     row = re.sub(r'[?]', ' ? ', row)
#     row = re.sub(r'[#]', ' # ', row)
#     row = re.sub(r'[@]', ' @ ', row)
#     row = re.sub(r'[$%^&*(){}=_;:"“”‘’.>,<`~.\\\[\]\-΄+«»]', ' ', row)
#     row = re.sub(r'[🧡☀️🥤🤷‍♂☕🤭🤣❤🥓🤡☘⚽🤔🤩♀🥄✌🤦‍♀€🤓🥳🤪🤮🤬♥☎☘⭐🤤❣®™🤗🥰💋💕📌😳🥺🖐🏿🍿🌱🍔🎶⛔🤏🤨]', ' ', row)
#     row = re.sub(r"[']", ' ', row)
#     row = re.sub(r"[⁦⁩]", ' ', row)
#     row = re.sub(r"[/]", ' ', row)
#     row = re.sub(r"\t", " ", row)
#     row = re.sub(r"'\s+\s+'", " ", row)
#     row = re.sub(r" ️", "", row)
#     row = re.sub(r'\b\w{1,1}\b', '', row)
#     row = emoji_pattern.sub(r'', row)
#     row = re.sub(' +', ' ', row)
#     return row


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def clear_text_simple(row_raw):
    # row_raw = re.sub(r'RT+ @\s*\S+', ' ', row_raw)
    # row_raw = re.sub(r'rt+ @\s*\S+', ' ', row_raw)
    row_raw = re.sub(r'RT+ @', '@ ', row_raw)
    row_raw = re.sub(r'rt+ @', '@ ', row_raw)
    row_lowered = row_raw.lower()
    row_split = row_lowered.split()
    tokens = [token for token in row_split
              if not token.startswith('http')
              ]
    row = TreebankWordDetokenizer().detokenize(tokens)
    row = re.sub('\n', ' ', row)
    # row = re.sub(r'(@\S+)', ' ', row)
    row = re.sub(r"\t", " ", row)
    row = re.sub(r'@', ' @ ', row)
    row = re.sub(r'#', ' # ', row)
    row = re.sub(r'\$', ' $ ', row)
    row = re.sub(r'%', ' % ', row)
    row = re.sub(r'\^', ' ^ ', row)
    row = re.sub(r'&', ' & ', row)
    row = re.sub(r'\?', ' ? ', row)
    row = re.sub(r'\*', ' * ', row)
    row = re.sub(r'!', ' ! ', row)
    row = re.sub(r'\.', ' . ', row)
    row = re.sub(r',', ' , ', row)
    row = re.sub(r'\(', ' ( ', row)
    row = re.sub(r'\)', ' ) ', row)
    row = re.sub(r'{', ' { ', row)
    row = re.sub(r'}', ' } ', row)
    row = re.sub(r'=', ' = ', row)
    row = re.sub(r';', ' ; ', row)
    row = re.sub(r':', ' : ', row)
    row = re.sub(r'"', ' " ', row)
    row = re.sub(r'\+', ' + ', row)
    row = re.sub(r'/', ' / ', row)
    row = re.sub(r'[“”‘><`~\\\[\]\΄«»…]', ' ', row)  # ’
    row = re.sub(r"'\s+\s+'", " ", row)
    # row = re.sub(r" ️", "", row)
    # row = re.sub(' +', " ", row)
    # row = re.sub(r'\b\w{1,1}\b', '', row)
    return row


def clear_text_very_simple(row_raw):
    row_raw = re.sub(r'RT+ @', '@ ', row_raw)
    row_raw = re.sub(r'rt+ @', '@ ', row_raw)
    row_lowered = row_raw.lower()
    row_split = row_lowered.split()
    tokens = [token for token in row_split
              if not token.startswith('http')
              ]
    row = TreebankWordDetokenizer().detokenize(tokens)
    row = re.sub('\n', ' ', row)
    row = re.sub(r"'\s+\s+'", " ", row)
    row = re.sub(' +', " ", row)
    return row


def remoce_accent_method(data, accent_col, new_accent):
    data[new_accent] = data.apply(lambda row: strip_accents(row[accent_col]), axis=1)
    return data


def clear_text_simple_method(data, text_col, new_text):
    data[new_text] = data.apply(lambda row: clear_text_simple(row[text_col]), axis=1)
    return data


# def clear_text_whole_method(data, text_col, new_text):
#     data[new_text] = data.apply(lambda row: clear_text_whole(row[text_col]), axis=1)
#     return data


def replace_method(row, target):
    target_list = [target, '@' + target, '#' + target]
    token_list = []
    row = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                 ' [URL] ', row)
    row = re.sub(r'[🧡☀️🥤🤷‍♂☕🤭🤣❤🥓🤡☘⚽🤔🤩♀🥄✌🤦‍♀€🤓🥳🤪🤮🤬♥☎☘⭐🤤❣®™🤗🥰💋💕📌😳🥺🖐🏿🍿🌱🍔🎶⛔🤏🤨]',
                 ' [EMOTICON] ', row)
    row_split = row.split()
    for token in row_split:
        if token not in target_list and token[0] == '@':
            token = '[USERNAME]'
        token_list.append(token)
    row = TreebankWordDetokenizer().detokenize(token_list)
    row = re.sub(r'RT+', ' ', row)
    row = re.sub(r'#', ' # ', row)
    row = re.sub(' +', ' ', row)
    return row


def replace(data, col, col_target, new_col):
    data[new_col] = data.apply(lambda row: replace_method(row[col], row[col_target]), axis=1)
    return data


def adding_spaces_method(row_raw):
    row_raw = re.sub(r'RT+ @', '@ ', row_raw)
    row_raw = re.sub(r'rt+ @', '@ ', row_raw)
    row_split = row_raw.split()
    tokens = [token for token in row_split
              if not token.startswith('http')
              ]
    row = TreebankWordDetokenizer().detokenize(tokens)
    row = re.sub('\n', ' ', row)
    row = re.sub(r"\t", " ", row)
    row = re.sub(r'@', ' @ ', row)
    row = re.sub(r'#', ' # ', row)
    row = re.sub(r'\$', ' $ ', row)
    row = re.sub(r'%', ' % ', row)
    row = re.sub(r'\^', ' ^ ', row)
    row = re.sub(r'&', ' & ', row)
    row = re.sub(r'\?', ' ? ', row)
    row = re.sub(r'\*', ' * ', row)
    row = re.sub(r'!', ' ! ', row)
    row = re.sub(r'\.', ' . ', row)
    row = re.sub(r',', ' , ', row)
    row = re.sub(r'\(', ' ( ', row)
    row = re.sub(r'\)', ' ) ', row)
    row = re.sub(r'{', ' { ', row)
    row = re.sub(r'}', ' } ', row)
    row = re.sub(r'=', ' = ', row)
    row = re.sub(r';', ' ; ', row)
    row = re.sub(r':', ' : ', row)
    row = re.sub(r'"', ' " ', row)
    row = re.sub(r'\+', ' + ', row)
    row = re.sub(r'/', ' / ', row)
    row = re.sub(r'[“”‘><`~\\\[\]\΄«»…]', ' ', row)  # ’
    row = re.sub(r"'\s+\s+'", " ", row)
    return row


def adding_spaces(data, text_col, new_text):
    data[new_text] = data.apply(lambda row: adding_spaces_method(row[text_col]), axis=1)
    return data


# nltk.download('stopwords')

# stop_words_el = stopwordsiso.stopwords(["el"])
# ok_stop_words_el = ['δεν', 'μη', 'κακώς', 'κακως', 'καλώς', 'καλως']
# stop_words_en = set(stopwords.words('english'))
# ok_stop_words_en = ['not', 'but', 'won']

# ok_stop_words = [*ok_stop_words_el, *ok_stop_words_en]

# stop_words = [*stop_words_el, *stop_words_en]

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def clear_text_whole(row_raw):
    row_raw = re.sub(r'RT+ @\s*\S+', ' ', row_raw)
    row_raw = re.sub(r'rt+ @\s*\S+', ' ', row_raw)

    row_lowered = row_raw.lower()
    row_split = row_lowered.split()
    tokens = [token for token in row_split
              if not token.startswith('http')
              ]
    row = TreebankWordDetokenizer().detokenize(tokens)
    row = re.sub(r'[!]', ' ', row)
    row = re.sub(r'[?]', ' ', row)
    row = re.sub(r'[#]', ' ', row)
    row = re.sub(r'[@]', ' ', row)
    row = re.sub(r'[$%^&*(){}=_;:"“”‘’.>,<`~.\\\[\]\-΄+«»]', ' ', row)
    row = re.sub(r'[🧡☀️🥤🤷‍♂☕🤭🤣❤🥓🤡☘⚽🤔🤩♀🥄✌🤦‍♀€🤓🥳🤪🤮🤬♥☎☘⭐🤤❣®™🤗🥰💋💕📌😳🥺🖐🏿🍿🌱🍔🎶⛔🤏🤨]', ' ', row)
    row = re.sub(r"[']", ' ', row)
    row = re.sub(r"[⁦⁩]", ' ', row)
    row = re.sub(r"[/]", ' ', row)
    row = re.sub(r"\t", " ", row)
    row = re.sub(r"'\s+\s+'", " ", row)
    row = re.sub(r"\t", " ", row)
    row = re.sub(r'@', ' ', row)
    row = re.sub(r'#', ' ', row)
    row = re.sub(r'\$', ' ', row)
    row = re.sub(r'%', ' ', row)
    row = re.sub(r'\^', ' ', row)
    row = re.sub(r'&', ' ', row)
    row = re.sub(r'\?', ' ', row)
    row = re.sub(r'\*', ' ', row)
    row = re.sub(r'!', ' ', row)
    row = re.sub(r'\.', ' ', row)
    row = re.sub(r',', ' ', row)
    row = re.sub(r'\(', ' ', row)
    row = re.sub(r'\)', ' ', row)
    row = re.sub(r'{', ' ', row)
    row = re.sub(r'}', ' ', row)
    row = re.sub(r'=', ' ', row)
    row = re.sub(r';', ' ', row)
    row = re.sub(r':', ' ', row)
    row = re.sub(r'"', ' ', row)
    row = re.sub(r'\+', ' ', row)
    row = re.sub(r'/', ' ', row)
    row = re.sub(r'[“”‘><`~\\\[\]\΄«»…]', ' ', row)  # ’
    row = re.sub(r"'\s+\s+'", " ", row)
    row = re.sub(r'\b\w{1,1}\b', '', row)
    row = emoji_pattern.sub(r'', row)
    row = re.sub(' +', ' ', row)
    return row


def clear_text_whole_method(data, text_col, new_text):
    data[new_text] = data.apply(lambda row: clear_text_whole(row[text_col]), axis=1)
    return data
