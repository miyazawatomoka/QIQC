from unittest import TestCase
from dataset import Word2vecStaticDataset


class TestWord2vecStaticDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.label_path = '../../data/train_label.npy'
        self.data_path = '../../data/word2vec_martix.npy'
        self.train_wsd = Word2vecStaticDataset(label_path=self.label_path, data_path=self.data_path, is_train=True)
        self.test_wsd = Word2vecStaticDataset(label_path=self.label_path, data_path=self.data_path, is_train=False)

    def test_len(self):
        train_len = len(self.train_wsd)
        test_len = len(self.test_wsd)
        # print(train_len)
        # print(test_len)

    def test_get_item(self):
        tr = self.train_wsd[11]
        te = self.test_wsd[2]
        # print(tr)
        # print(te)