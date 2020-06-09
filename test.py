import tensorflow as tf
from tensorflow.python import keras
import pandas as pd
import jieba
import numpy as np
import time
from tensorflow.python.ops import nn_ops as nn


def DataGet(filename='./data.csv'):
    """ Data reading from disk """
    data = pd.read_csv(filename, header=None, encoding='utf-8')
    data = data.dropna().sample(frac=1)
    sentences = np.array(data[0])
    words_sequence = [
        list(jieba.cut(sentence)) for sentence in sentences
    ]

    UNK = '<UNK>'
    words_dict = {UNK: 0}

    data_in, data_out = [], []
    for sentence in words_sequence:
        for word in sentence:
            if word not in words_dict:
                words_dict[word] = len(words_dict)
        data_in.append([words_dict[word] for word in sentence])

    answer_labels = {}
    for answer in data[1]:
        if answer not in answer_labels:
            answer_labels[answer] = len(answer_labels)
        data_out.append(answer_labels[answer])

    labels = data[1].apply(lambda x: answer_labels[x])
    return data_in, data_out, words_dict, answer_labels


class Model(keras.Model):
    """
    DSSM-LSTM, version 1.
    Using Jieba to cut words, without tri-grams.
    Do softmax in the end, only apply the input layer
    and express layer on the queries.
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.embedding = keras.layers.Embedding(
            input_dim=input_dim, output_dim=embedding_dim)
        self.lstm = keras.layers.LSTM(
            hidden_dim)
        self.linear1 = keras.layers.Dense(
            256, activation='tanh')
        self.linear2 = keras.layers.Dense(
            128, activation='tanh')
        self.linear3 = keras.layers.Dense(
            64, activation=nn.softmax)

    def call(self, inputs):
        print(inputs)
        embeded = self.embedding(inputs)
        lstm_out = self.lstm(embeded)
        out1 = self.linear1(lstm_out)
        out2 = self.linear2(out1)
        return self.linear3(out2)


def input_fn(data_in, labels, batch_size=1, num_epoches=50):
    ds_in = np.array([np.array(din) for din in data_in])
    ds_out = np.array(labels)
    return ds_in, ds_out


if __name__ == "__main__":
    UNK = '<UNK>'
    start_tm = time.time()
    words_in, labels, words2idx, answer2idx = DataGet()
    print('data dictionary process using',
          time.time() - start_tm, 'sec')

    ds_in, ds_out = input_fn(words_in, labels)
    epoch_num = 100

    # model
    INPUT_DIM = len(words2idx)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(answer2idx)
    model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    model.compile(
        keras.optimizers.Adam(lr=0.001),
        keras.losses.SparseCategoricalCrossentropy,
        metrics=['accuracy']
    )

    print('start training model')
    model.fit(ds_in, ds_out, batch_size=1, epochs=epoch_num)
