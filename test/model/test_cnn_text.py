from unittest import TestCase
from model import TextCNN
import numpy as np
import torch
from torch.autograd import Variable


class TestCNNTest(TestCase):
    def test_model(self):
        model = TextCNN()
        # print(model)

    def test_forword(self):
        model = TextCNN()
        word2vec_martix = np.load('../data/word2vec_martix.npy')
        mt = torch.tensor(word2vec_martix)
        vmt = Variable(mt)
        mmt = model(vmt)
        # print(mmt)
