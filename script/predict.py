from utils import WordEmbeddingUtil, TextUtil
from config import Config
import numpy as np
import torch

word2vec_util = None
text_cnn_model = torch.load('../pretrained/text_cnn_static.h5')


def static_text_cnn_word2vec_predict(sentence):
    global word2vec_util, text_cnn_model
    if word2vec_util is None:
        word2vec_util = WordEmbeddingUtil()
    text_util = TextUtil()
    row = text_util.text_normalization(sentence)
    words = text_util.lemmatize_sentence(row)
    words = text_util.filter_punctuation(words)
    words = text_util.filter_stop_word(words)
    words = text_util.get_words_with_len(words)
    words_matrix = np.zeros([Config.SENTENCE_MAX_LEN, Config.EMBEDDING_SIZE], dtype=np.float32)
    for idx, word in enumerate(words):
        words_matrix[idx] = word2vec_util.get_word2vec_vec(word)
    text_cnn_model.eval()
    words_matrix_tensor = torch.Tensor(words_matrix)
    words_matrix_tensor = torch.unsqueeze(words_matrix_tensor, 0)
    predict = text_cnn_model(words_matrix_tensor)
    result = predict.item()
    return result


if __name__ == '__main__':
    print(static_text_cnn_word2vec_predict("hello world"))
