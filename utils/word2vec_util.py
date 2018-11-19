import gensim
from config import Config


class Word2vecUtil:
    def __init__(self, word2vec_path=Config.WORD2VEC_PATH):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def get_word2idx(self):
        words_list = list(self.word2vec.vocab.keys())
        word2idx = {}
        for idx, word in enumerate(words_list):
            word2idx[word] = idx
        return word2idx

    def get_weight(self):
        return self.word2vec.vectors
