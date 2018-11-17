from unittest import TestCase
from trainer import TextCNNTrainer
from dataset import Word2vecStaticDataset


class TestCNNTrainer(TestCase):

    def setUp(self):
        super().setUp()
        self.train_dataset = Word2vecStaticDataset(is_train=True,
                                                   label_path='../../data/train_label.npy',
                                                   data_path='../../data/word2vec_martix.npy')
        self.test_dataset = Word2vecStaticDataset(is_train=False,
                                                  label_path='../../data/train_label.npy',
                                                  data_path='../../data/word2vec_martix.npy')
        self.tct = TextCNNTrainer(train_dataset=self.test_dataset, test_dataset=self.test_dataset)

    def test_train(self):
        self.tct.train()

    def test_test(self):
        self.tct.test()
