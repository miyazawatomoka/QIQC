from unittest import TestCase
from model import TextCNN
import numpy as np
import torch
from torch.autograd import Variable
from utils import Word2vecUtil, WordEmbeddingUtil
from config import Config
import gc


class TestCNNTest(TestCase):
    def setUp(self):
        super().setUp()
        # self.weight = WordEmbeddingUtil().get_embedding_weight(Word2vecUtil(Config.WORD2VEC_PATH).get_weight())
        # self.model = TextCNN(pretrained_weight=self.weight )
        # gc.collect()

    def test_forword(self):
        pass
        # model = TextCNN()
        # tdata = np.load('../data/data_in_inx.npy')
        # mt = torch.LongTensor(tdata)
        # vmt = Variable(mt)
        # mmt = model(vmt)
        # # print(mmt)
