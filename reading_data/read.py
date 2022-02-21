import pandas as pd
import os


def raw_datasets(directory):
    col_names = ['id', 'createdate', 'channel', 'text', 'targets', 'sentiments']
    data_GK = pd.DataFrame(columns=col_names)
    data_GR = pd.DataFrame(columns=col_names)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.__contains__('GK'):
                path = directory + file
                data_GK_temp = pd.read_excel(path)
                data_GK = data_GK.append(data_GK_temp)
            elif file.__contains__('GR'):
                path = directory + file
                data_GR_temp = pd.read_excel(path)
                data_GR = data_GR.append(data_GR_temp)

    return data_GK, data_GR


def create_dataframe(data):
    df_fin = data.rename(columns={
        'channel_x': 'channel',
        'createdate_x': 'createdate',
        'text_x': 'text',
        'targets_x': 'targets_GK',
        'sentiments_x': 'sentiments_GK',
        'targets_y': 'targets_GR',
        'sentiments_y': 'sentiments_GR'
    })
    df_fin = df_fin.drop(columns=['createdate_y', 'channel_y', 'text_y'])
    print(list(df_fin.columns.values))
    return df_fin


def read_df_per_row_sentiment():
    path = 'data_custom/'
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(file)
    name = file_list[-1]
    filepath = path + name
    data = pd.read_csv(filepath,
                       sep=','
                       )
    data = data.drop(columns='Unnamed: 0')
    return data


def read_phase3():
    path = 'data_phase2/'
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(file)
    name = file_list[-1]
    filepath = path + name
    data = pd.read_csv(filepath,
                       sep=','
                       )
    data = data.drop(columns='Unnamed: 0')
    return data
