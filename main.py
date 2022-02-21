from reading_data.read import *
from preprocessing.preprocess import *
from preprocessing.create_datasets import *
from models.scikit_models import *
from models.rnn_models import *
from models.bert_models import *
from models.bert_rnn_models import *
from svm import *

ft = fasttext.load_model('fasttext/cc.el.300.bin')

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

SEED = 11


def step1():
    directory = 'data/'
    # read datasets from each annotator
    data_GK, data_GR = raw_datasets(directory)

    # remove extra rows
    data_GK = remove_empty_rows(data_GK, 'id')
    data_GR = remove_empty_rows(data_GR, 'id')

    # check if columns have errors
    data_GK, data_GR = check_if_cols_are_identical(data_GK, data_GR, check=True, show_errors=False)

    # check if targets are empty
    data_GK, data_GR = check_if_targets_are_NA(data_GK, data_GR)

    # remove rows where number of sentiments != number of targets
    # (since we do not know which sentiment applies to which target)
    data_GK, data_GR = discard_mismatch_in_sentiments_and_targets(data_GK, data_GR)
    data = keep_same_ids(data_GK, data_GR)
    data = create_dataframe(data)
    data = clear_target_col(data)
    # print(data.columns.values)

    # create final dataset with agreed upon sentiment and target
    data = keep_only_agreed_upon_sentiments(data)

    # just to be sure! (1 line used to pass through here..)
    data = drop_na_sentiments(data)  # 4580
    data.to_csv('test.csv')
    # unroll dataset
    data = unroll_sentiments(data)  # 6642
    data.to_csv('test2.csv')

    # save unrolled dataset
    save(data)


def step2():
    # read unrolled dataset
    data = read_df_per_row_sentiment()

    # delete \n character -> since it causes issues with FastText library
    data = one_line_pre(data)

    # solve imbalance
    data = imb_method(data)

    # create new columns for all variations
    data = create_new_columns_for_variations(data)

    # replace sentiment ranging [-2,2] to [0,4]
    data = replace_sentiment_range(data)

    # create datasets
    create_three_datasets(data)


def run_preprocess():
    step1()
    step2()


def run_methods():
    scikit_methods()    # SVM, MLP
    rnn_methods()       # LSTM, BiLSTM, GRU, BiGRU
    bert_methods()      # BERT pre-trained
    bert_gru_methods()  # BERT + GRU layer + SUM / CONC last 4 hidden layers


if __name__ == '__main__':
    run_preprocess()
    run_methods()
