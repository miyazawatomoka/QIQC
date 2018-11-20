import sys
sys.path.append('../')
from utils import WordEmbeddingUtil, TextUtil, Word2vecUtil
from config import Config
import numpy as np
import torch
import pandas as pd
import tqdm
import gc


word2vec_util = None
text_cnn_model = torch.load('../pretrained/text_cnn_static.h5')
df = pd.read_csv('../../input/test.csv')
model = torch.load('../../pretrained/text_cnn_static.h5')


def cnn_predict():
    text_util = TextUtil()
    # text normalization
    question_text_np_array = df.question_text.values
    for idx, row in enumerate(tqdm.tqdm(question_text_np_array)):
        question_text_np_array[idx] = text_util.text_normalization(row)
    # stem words 词性还原
    stem_words = []
    for row in tqdm.tqdm(question_text_np_array):
        words = text_util.lemmatize_sentence(row)
        stem_words.append(words)
    # 去除标点与停用词
    filter_words = []
    for words in tqdm.tqdm(stem_words):
        words = text_util.filter_punctuation(words)
        words = text_util.filter_stop_word(words)
        filter_words.append(words)
    pad_and_cut_words = []
    for words in tqdm.tqdm(filter_words):
        if len(words) < Config.SENTENCE_MAX_LEN - 2:
            words = [Config.START_CHAR] + words + [Config.END_CHAR] + [Config.PADDING_CHAR] * (
                        Config.SENTENCE_MAX_LEN - 2 - len(words))
        else:
            words = [Config.START_CHAR] + words[0: Config.SENTENCE_MAX_LEN - 2] + [Config.END_CHAR]
        pad_and_cut_words.append(words)
    word2vec_util = Word2vecUtil(Config.WORD2VEC_PATH)
    word_embedding_util = WordEmbeddingUtil()
    word2idx = word2vec_util.get_word2idx()
    # weight = Word2vecUtil.get_weight()
    gc.collect()
    tdata = np.zeros([len(pad_and_cut_words), Config.SENTENCE_MAX_LEN], dtype=np.int64)
    for i, words in enumerate(pad_and_cut_words):
        for j, word in enumerate(words):
            tdata[i][j] = word_embedding_util.get_idx_by_word(word2idx, word)
    np.save('../../cache/word2vec_idx_test.npy', tdata)
    # 训练
    vdata = torch.LongTensor(tdata)
    vdata = vdata.cuda()
    with torch.no_grad():
        i = 0
        l = len(vdata)
        result = np.zeros(0)
        while i < l:
            p = vdata[i:i + 1000]
            i += 1000
            pre = model(p)
            pre = pre.cpu().numpy()
            result = np.append(result, pre)
            torch.cuda.empty_cache()
    np.save('../../cache/word2vec_result.npy', result)


if __name__ == '__main__':
    cnn_predict()
