#!pip install transformers

import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import os

import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from transformers import AutoTokenizer #, AutoModel
from transformers import TFAutoModel
from transformers import BertConfig

def tokenizer_method(col, seq_len):
    tokens = tokenizer(col.tolist(),
                       max_length=seq_len,
                       truncation=True,
                       padding='max_length',
                       add_special_tokens=True,
                       return_tensors='np')  # tf, pt
    return tokens


def array_create(text_array, target_array, col, add_separator):
    tokens = text_array[col]
    tokens = np.roll(tokens, 17)
    for i, row in enumerate(tokens):
        tokens[i][0:16] = target_array[col][i]
        if add_separator:
            tokens[i][16] = 102

    return tokens


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids,
            'attention_mask': masks}, labels


def prep_data(text, target):
    tokens_new_text = tokenizer.encode_plus(text, max_length=seq_len,
                                            truncation=True, padding='max_length',
                                            add_special_tokens=True, return_token_type_ids=False,
                                            return_tensors='tf')
    tokens_new_target = tokenizer.encode_plus(target, max_length=target_seq_len,
                                              truncation=True, padding='max_length',
                                              add_special_tokens=True, return_token_type_ids=False,
                                              return_tensors='tf')

    tokens = array_create(tokens_new_text, tokens_new_target, 'input_ids', True)
    mask = array_create(tokens_new_text, tokens_new_target, 'attention_mask', False)

    # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
    return {'input_ids': tf.cast(tokens, tf.float64),
            'attention_mask': tf.cast(mask, tf.float64)}


def add_sent_value(pred):
    return pred


def bert_gru_methods():
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

    config = BertConfig.from_pretrained("nlpaueb/bert-base-greek-uncased-v1",
                                        output_hidden_states=True)

    for i, item in enumerate(train_list):
        #
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", config=config)
        bert = TFAutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", config=config)
        #
        df = pd.read_csv(train_list[0], sep=',')
        df_test = pd.read_csv(test_list[0], sep=',')
        #
        # print(item)
        print(train_list[1])
        print(len(df))
        print(test_list[1])
        print(len(df_test))
        #
        # m_name = dataset_train[37:45]
        seq_len = 512
        target_seq_len = 16
        batch_size = 64
        split = 0.9
        #
        tokens_text = tokenizer_method(df['text'], seq_len)
        tokens_target = tokenizer_method(df['target'], target_seq_len)
        #
        tokens = array_create(tokens_text, tokens_target, 'input_ids', True)
        mask = array_create(tokens_text, tokens_target, 'attention_mask', False)
        #
        arr = df['sentiment'].values
        num_samples = len(df)
        labels = np.zeros((num_samples, arr.max() + 1))
        labels[np.arange(num_samples), arr] = 1
        #
        with open('..//bert_arrays/tokens.npy', 'wb') as f:
            np.save(f, tokens)
        with open('..//bert_arrays/mask.npy', 'wb') as f:
            np.save(f, mask)
        with open('..//bert_arrays/sentiments.npy', 'wb') as f:
            np.save(f, labels)
        with open('..//bert_arrays/tokens.npy', 'rb') as f:
            Xids = np.load(f, allow_pickle=True)
        with open('..//bert_arrays/mask.npy', 'rb') as f:
            Xmask = np.load(f, allow_pickle=True)
        with open('..//bert_arrays/sentiments.npy', 'rb') as f:
            labels = np.load(f, allow_pickle=True)
        #
        dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
        #
        dataset = dataset.map(map_func)
        dataset = dataset.batch(batch_size, drop_remainder=True)  # .shuffle(10000)
        size = int(Xids.shape[0] / batch_size * split)
        #
        train_ds = dataset.take(size)
        val_ds = dataset.skip(size)
        tf.data.experimental.save(train_ds, 'train')
        tf.data.experimental.save(val_ds, 'val')
        ds = tf.data.experimental.load('train', element_spec=train_ds.element_spec)
        #
        # inputs
        input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

        # 12 hidden state embeddings
        embeddings = bert.bert(input_ids, attention_mask=mask)

        # Concatenate last 4 hidden layers
        # conc_embeddings = tf.keras.layers.Concatenate(axis=1)([embeddings[2][-1],
        #                                                        embeddings[2][-2],
        #                                                        embeddings[2][-3],
        #                                                        embeddings[2][-4]])

        # Sum last 4 hidden layers
        summed_embeddings = embeddings[2][-1] + embeddings[2][-2] + embeddings[2][-3] + embeddings[2][-4]

        #
        # # classifier head
        # x1 = tf.keras.layers.LSTM(512, dropout=.2, recurrent_dropout=.2, return_sequences=True)(embeddings)
        # x2 = tf.keras.layers.LSTM(512, dropout=.2, recurrent_dropout=.2, return_sequences=False)(x1)
        #
        x2 = tf.keras.layers.GRU(512, dropout=.2, recurrent_dropout=.2)(summed_embeddings)
        # normalize
        x3 = tf.keras.layers.BatchNormalization()(x2)
        # output
        # x4 = tf.keras.layers.Dense(515, activation='relu')(x3)
        x5 = tf.keras.layers.Dense(256, activation='relu')(x3)
        y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x5)
        #
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
        # model.layers[2].trainable = False
        #
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)  # 1e-3 5e-5
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        #
        model.compile(optimizer=optimizer, loss=loss, metrics=[acc])
        #
        element_spec = ({'attention_mask': tf.TensorSpec(shape=(batch_size, 512), dtype=tf.int64, name=None),
                         'input_ids': tf.TensorSpec(shape=(batch_size, 512), dtype=tf.int64, name=None)},
                        tf.TensorSpec(shape=(batch_size, 5), dtype=tf.float64, name=None))
        #
        train_ds = tf.data.experimental.load('train', element_spec=element_spec)
        val_ds = tf.data.experimental.load('val', element_spec=element_spec)
        #
        for i in range(1, 7):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1
            )
        #
        sent_pred = []
        leng = len(df_test)
        for i, row in df_test.iterrows():
            tokens = prep_data(row['text'], row['target'])
            probs = model.predict(tokens)
            pred = np.argmax(probs)
            sent_pred.append(pred)
        #
        gold_labels = list(df_test['sentiment'])
        #
        print(metrics.accuracy_score(gold_labels, sent_pred))
        print(metrics.precision_score(gold_labels, sent_pred, average="macro"))
        print(metrics.recall_score(gold_labels, sent_pred, average="macro"))
        print(metrics.f1_score(gold_labels, sent_pred, average="macro"))
        print(metrics.f1_score(gold_labels, sent_pred, average="micro"))
        print(metrics.f1_score(gold_labels, sent_pred, average="weighted"))
        print(metrics.classification_report(gold_labels, sent_pred))

        model.save('..//content/gdrive/MyDrive/models/BERT_C4L.h5')

        cf_matrix = confusion_matrix(gold_labels, sent_pred)
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

        dir = '..//content/gdrive/MyDrive/figures/'
        name = time.strftime("%Y%m%d-%H%M%S")
        path = dir + name + '.png'

        plt.savefig(path, dpi=600)
        plt.show()
