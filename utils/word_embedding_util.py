from config import Config
import pickle
import os
import gensim
import numpy as np


class WordEmbeddingUtil:
    WORD_TO_FREQUENCY_CACHE_PATH = "../cache/word_to_frequency.pkl"

    def __init__(self, word2vec_path=Config.WORD2VEC_PATH):
        if word2vec_path:
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.unk_vec = np.random.rand(Config.EMBEDDING_SIZE)

    @staticmethod
    def get_vovab(words_list, word_frequency=Config.WORD_FREQUENCY):
        vovab = [Config.START_CHAR, Config.END_CHAR, Config.PADDING_CHAR, Config.UNKNOWN_CHAR]
        word_to_frequency = WordEmbeddingUtil.get_word_frequency(words_list=words_list)
        for word, frequency in word_to_frequency.items():
            if frequency > word_frequency:
                vovab.append(word)
        return vovab

    @staticmethod
    def get_word_frequency(words_list, in_path=None, out_path=None):
        if in_path is not None and os.path.exists(in_path):
            with open(in_path, mode='rb') as in_f:
                return pickle.load(in_f)
        word_to_frequency = {}
        for words in words_list:
            for word in words:
                if word in word_to_frequency:
                    word_to_frequency[word] += 1
                else:
                    word_to_frequency[word] = 1
        if out_path is not None:
            with open(out_path, mode='wb') as out_f:
                pickle.dump(obj=word_to_frequency, file=out_f)
        return word_to_frequency

    def get_word2vec_vec(self, word):
        if word == Config.PADDING_CHAR:
            return np.zeros(Config.EMBEDDING_SIZE, dtype=np.float32)
        elif word == Config.START_CHAR:
            return np.ones(Config.EMBEDDING_SIZE, dtype=np.float32)
        elif word == Config.END_CHAR:
            return np.full(Config.EMBEDDING_SIZE, -1, dtype=np.float32)
        elif not (self.is_word_in_word2vec(word) and self.is_word_in_word2vec(word.lower())):
            return self.unk_vec
        elif (not self.is_word_in_word2vec(word)) and self.is_word_in_word2vec(word.lower()):
            word = word.lower()
        return self.word2vec[word]

    def is_word_in_word2vec(self, word):
        return word in self.word2vec.vocab.keys()

