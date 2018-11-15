from unittest import TestCase
from utils import TextUtil


class TestTextUtil(TestCase):
    def setUp(self):
        super().setUp()
        self.text_util = TextUtil()

    def test_text_normalization(self):
        in1 = "I'll eat an apple."
        expect1 = "I will eat an apple."
        out1 = self.text_util.text_normalization(in1)
        self.assertEqual(expect1, out1)

    def test_filter_stop_word(self):
        in1 = ['I', 'am', 'panda']
        expect1 = ['I', 'panda']
        out1 = self.text_util.filter_stop_word(in1)
        self.assertListEqual(out1, expect1)

    def test_filter_punctuation(self):
        in1 = ['Because', ',', 'I', 'am', 'panda', '.']
        expect1 = ['Because', 'I', 'am', 'panda']
        out1 = self.text_util.filter_punctuation(in1)
        self.assertListEqual(out1, expect1)