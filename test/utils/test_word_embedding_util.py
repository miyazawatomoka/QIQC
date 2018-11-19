from unittest import TestCase
from utils import WordEmbeddingUtil
from config import Config
import numpy as np


class TestWordEmbeddingUtil(TestCase):
    def setUp(self):
        super().setUp()
        self.word_embedding_util = WordEmbeddingUtil()

    def test_get_vovab(self):
        in1 = [['I'], ['am', 'panda']]
        out1 = WordEmbeddingUtil.get_vovab(in1, 0)
        expect1 = [Config.START_CHAR, Config.END_CHAR, Config.PADDING_CHAR, Config.UNKNOWN_CHAR]
        expect1 += ['I', 'am', 'panda']
        self.assertListEqual(sorted(out1), sorted(expect1))

    def test_get_word_frequency(self):
        pass

    def test_get_embedding_weight(self):
        weight1 = np.zeros([3, Config.EMBEDDING_SIZE], dtype=np.float32)
        wordembedding_util = WordEmbeddingUtil()
        result = wordembedding_util.get_embedding_weight(weight1)
        self.assertEqual(result.shape[0], 7)
        self.assertEqual(result.shape[1], Config.EMBEDDING_SIZE)

    def test_get_idx_by_word(self):
        in1 = "Hello"
        in2 = "World"
        word2idx = {'Hello': 0}
        out1 = self.word_embedding_util.get_idx_by_word(word2idx=word2idx, word=in1)
        out2 = self.word_embedding_util.get_idx_by_word(word2idx=word2idx, word=in2)
        expect1 = 4
        self.assertEqual(out1, expect1)
        self.assertEqual(out2, Config.UNKNOWN_IDX)