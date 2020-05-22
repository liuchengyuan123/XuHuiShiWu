import jieba.analyse
import os
import jieba
import numpy as np
import torch
import torch.nn as nn


def DataPrepare(que_seq, word2idx):
    """ Return shape (len(que_seq)) """
    ret = [word2idx[word] for word in que_seq]
    return torch.LongTensor(ret)


class TransAtten(nn.Module):

    """
    A model similar with the encoder of Transformer Model.
    But in this part, no positional encoding.

    Word embedding; `query`, `key` and `value` matrix; output softmax.
    """

    def __init__(self, input_size, embedding_dim, qk_dim, output_dim):
        super(TransAtten, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.query = nn.Linear(embedding_dim, qk_dim)
        self.key = nn.Linear(embedding_dim, qk_dim)
        self.value = nn.Linear(embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence_in):
        len_seq = len(sequence_in)
        embeded = self.embedding(sequence_in.view(1, len_seq))
        Q = self.query(embeded).view(len_seq, -1)
        K = self.key(embeded).view(len_seq, -1)
        V = self.value(embeded).view(len_seq, -1)
        s = Q.matmul(K.T)
        s = self.softmax(s)
        output = s.matmul(V)
        return torch.sum(output, axis=0).view(1, -1) / len_seq


def RandomChoice(total_data, word2idx):
    sel = np.random.randint(low=0, high=len(total_data))
    sel = total_data[sel]
    x = DataPrepare(sel[0], word2idx)
    y = torch.LongTensor([sel[1]])
    return x, y


def train(total_data, word2idx,
          model, optimizer, criterion):
    model.zero_grad()
    seq_in, label = RandomChoice(total_data, word2idx)
    pred = model(seq_in)
    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()

    prediction = torch.argmax(pred)
    acc = int(prediction.item() == label.item())
    return loss.item(), acc


def PrePorcess():
    """ Pre process the csv data from raw csv files. """

    dir_name = './csv_data/'
    files = os.listdir(dir_name)
    question_answer = []
    for filename in files:
        whole = dir_name + filename
        with open(whole, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                question_answer.append([
                    u.strip() for u in line.split(',')[:2]])
                # t = [u.strip() for u in line.split(',')]
                # assert(len(t) == 2)

    # for further convenient usage
    question_answer = np.array(question_answer, dtype=object)

    """ # Unknown key word used as `<UNK>`
    word2idx = {}
    word2idx['<UNK>'] = 0

    for sentence in question_answer[:, 0]:
        for word in jieba.cut(sentence):
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    answer_set = list(set(question_answer[:, 1]))
    answer2idx = {'<UNA>': 0}
    for answer in answer_set:
        answer2idx[answer] = len(answer2idx) """

    answer2idx = np.load('./TransAtten_Sol1/anwser2idx.npy',
                         allow_pickle=True).item()
    word2idx = np.load('./TransAtten_Sol1/word2idx.npy',
                       allow_pickle=True).item()

    total_data = []
    for qa_pair in question_answer:
        tmp = []
        tmp.append(list(jieba.cut(qa_pair[0])))
        tmp.append(answer2idx[qa_pair[1]])
        total_data.append(tmp)
    total_data = np.array(total_data)
    # del answer_set, question_answer

    np.random.shuffle(total_data)
    return word2idx, answer2idx, total_data


if __name__ == '__main__':
    word2idx, answer2idx, total_data = PrePorcess()

    """
    # Because of not enough data,
    # we are not going to do this for a better performence
    train_rate = 0.8
    train_len = int(len(total_data) * train_rate)
    """

    # Model Settings
    EMBEDDING_DIM = 64
    QK_DIM = 128

    model = TransAtten(len(word2idx), EMBEDDING_DIM,
                       QK_DIM, len(answer2idx))
    model.load_state_dict(torch.load('./TransAtten_Sol1/model.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    display_epoch = 1000
    loss_sum, acc_sum = 0, 0
    epoch = 0
    goon = True
    while goon:
        epoch += 1
        loss, acc = train(total_data, word2idx,
                          model, optimizer, criterion)
        loss_sum += loss
        acc_sum += acc
        if epoch % display_epoch is 0:
            print(f'epoch {epoch}, average loss',
                  f'{loss_sum / display_epoch: .3f}, accuracy',
                  f'{acc_sum / display_epoch * 100:.1f} %')
            loss_sum, acc_sum = 0, 0

    print('terminate')
