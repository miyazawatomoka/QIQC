from unittest import TestCase
from utils import WordEmbeddingUtil
from config import Config


class TestWordEmbeddingUtil(TestCase):
    def test_get_vovab(self):
        in1 = [['I'], ['am', 'panda']]
        out1 = WordEmbeddingUtil.get_vovab(in1, 0)
        expect1 = [Config.START_CHAR, Config.END_CHAR, Config.PADDING_CHAR, Config.UNKNOWN_CHAR]
        expect1 += ['I', 'am', 'panda']
        self.assertListEqual(sorted(out1), sorted(expect1))

    def test_get_word_frequency(self):
        pass
