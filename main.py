from preprocessing.initiate_preprocess_steps import *
from models.scikit_models import *
from models.rnn_models import *
from models.bert_models import *
from models.bert_rnn_models import *

ft = fasttext.load_model('fasttext/cc.el.300.bin')

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


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
