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

def input_fn(data_in, labels, shuffle=True, batch_size=1, num_epoches=1):
    def generator():
        for d, l in zip(data_in, labels):
            yield (d, [l])
    data = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32))
    if shuffle:
        data = data.shuffle(1000)
    data = data.batch(batch_size).repeat(num_epoches)
    return data

if __name__ == "__main__":
    UNK = '<UNK>'
    start_tm = time.time()
    words_in, labels, words2idx, answer2idx = DataGet()
    data = input_fn(words_in, labels)

    INPUT_DIM = len(words2idx)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(answer2idx)
    model = keras.Sequential([
        keras.layers.Embedding(INPUT_DIM, EMBEDDING_DIM),
        keras.layers.LSTM(HIDDEN_DIM),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(OUTPUT_DIM, activation=nn.softmax)
    ])

    model.compile(
        keras.optimizers.Adam(lr=0.001),
        keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(data, epochs=50)
    # Very hard to train!!!
