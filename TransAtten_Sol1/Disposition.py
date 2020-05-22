import jieba
import torch
import numpy as np
from train import TransAtten


class Disposit:
    """
    Using `TransAtten_Sol1.train.TransAtten model to predict`
    """
    def __init__(self):
        self.word2idx = np.load('./TransAtten_Sol1/word2idx.npy',
                                allow_pickle=True).item()
        self.anwser2idx = np.load('./TransAtten_Sol1/anwser2idx.npy',
                                  allow_pickle=True).item()
        self.idx2anwser = {}
        for anwser in self.anwser2idx:
            self.idx2anwser[self.anwser2idx[anwser]] = anwser

        print(len(self.idx2anwser), len(self.anwser2idx))

        # Model Settings
        EMBEDDING_DIM = 64
        QK_DIM = 128
        self.model = TransAtten(len(self.word2idx), EMBEDDING_DIM,
                                QK_DIM, len(self.anwser2idx))
        self.model.load_state_dict(torch.load('./TransAtten_Sol1/model.pth'))
        del self.anwser2idx

    def react2anwser(self, anwseridx):
        anwser = self.idx2anwser[anwseridx]
        if anwser == '<UNA>':
            return "Unkown Anwser"
        return anwser

    def react2word(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx['<UNK>']

    def sen2seq(self, que_seq):
        ret = [self.react2word(word) for word in que_seq]
        return torch.LongTensor(ret)

    def precidt(self, question):
        test_in = self.sen2seq(jieba.cut(question))
        with torch.no_grad():
            label = self.model(test_in)
            return self.react2anwser(torch.argmax(label).item())
