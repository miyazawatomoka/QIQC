
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
import pandas as pd
from utils import TextUtil, WordEmbeddingUtil
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pickle
import tqdm
from config import Config
import numpy as np
from utils import Word2vecUtil
import gc


# In[2]:
# df = pd.read_excel('../data/train_sub.xlsx')
df = pd.read_csv('../data/train.csv')
# df = pd.read_csv('../../input/test.csv')


# In[3]:
text_util = TextUtil()


# In[4]:
# get train label
np.save('../data/train_label.npy', df.target.values)


# In[5]:
# text normalization
question_text_np_array = df.question_text.values
for idx, row in enumerate(tqdm.tqdm(question_text_np_array)):
    question_text_np_array[idx] = text_util.text_normalization(row)


# In[6]:
# stem words 词性还原
stem_words = []
for row in tqdm.tqdm(question_text_np_array):
    words = text_util.lemmatize_sentence(row)
    stem_words.append(words)


# In[7]:
# 去除标点与停用词
filter_words = []
for words in tqdm.tqdm(stem_words):
    words = text_util.filter_punctuation(words)
    words = text_util.filter_stop_word(words)
    filter_words.append(words)


# 预处理完的数据, 为如下的格式
# ```
# [['I', 'pandas'], ['How', 'Quebec', 'nationalist', 'see', 'province', 'nation']]
# ```

# In[8]:
pad_and_cut_words = []
for words in tqdm.tqdm(filter_words):
    if len(words) < Config.SENTENCE_MAX_LEN-2:
        words = [Config.START_CHAR] + words + [Config.END_CHAR] + [Config.PADDING_CHAR] * (Config.SENTENCE_MAX_LEN - 2 - len(words))
    else:
        words = [Config.START_CHAR] + words[0: Config.SENTENCE_MAX_LEN - 2] + [Config.END_CHAR]
    pad_and_cut_words.append(words)


# In[27]:
with open('../cache/words.pkl', mode='wb') as f:
    pickle.dump(file=f, obj=pad_and_cut_words)


# In[2]:
# with open('../cache/words.pkl', mode='rb') as f:
#     pad_and_cut_words = pickle.load(file=f)


# In[3]:
word2vec_util = Word2vecUtil(Config.WORD2VEC_PATH)
word_embedding_util = WordEmbeddingUtil()
word2idx = word2vec_util.get_word2idx()
# weight = Word2vecUtil.get_weight()
gc.collect()
tdata = np.zeros([len(pad_and_cut_words), Config.SENTENCE_MAX_LEN], dtype=np.int64)
for i, words in enumerate(pad_and_cut_words):
    for j, word in enumerate(words):
        tdata[i][j] = word_embedding_util.get_idx_by_word(word2idx, word)


# For Train
np.save('../../input/word2vec_idx_data.npy', tdata)

# For Test
# np.save(../../input/word2vec_test_data.npy', tdata)

