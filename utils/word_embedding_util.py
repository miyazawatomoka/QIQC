from config import Config
import pickle
import os
import gensim
import numpy as np


class WordEmbeddingUtil:
    WORD_TO_FREQUENCY_CACHE_PATH = "../cache/word_to_frequency.pkl"

    def __init__(self, word2vec_path=Config.WORD2VEC_PATH):
        self.unk_vec = np.random.rand(Config.EMBEDDING_SIZE)
        self.pad_vec = np.zeros(Config.EMBEDDING_SIZE, dtype=np.float32)
        self.start_vec = np.ones(Config.EMBEDDING_SIZE, dtype=np.float32)
        self.end_vec = np.full(Config.EMBEDDING_SIZE, -1, dtype=np.float32)

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

    @DeprecationWarning
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

    def get_embedding_weight(self, pretrained_weight):
        retain_chars = np.array([self.start_vec, self.end_vec, self.pad_vec, self.unk_vec])
        return np.vstack((retain_chars, pretrained_weight))

    @classmethod
    def get_idx_by_word(cls, word2idx, word):
        if word == Config.START_CHAR:
            return Config.START_IDX
        elif word == Config.END_CHAR:
            return Config.END_IDX
        elif word == Config.PADDING_CHAR:
            return Config.PADDING_IDX
        elif (word not in word2idx) and (word.lower() not in word2idx):
            return Config.UNKNOWN_IDX
        elif word in word2idx:
            return word2idx[word] + Config.RETAIN_COUNT
        elif (word not in word2idx) and (word.lower() in word2idx):
            return word2idx[word.lower()] + Config.RETAIN_COUNT




