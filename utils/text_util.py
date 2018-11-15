import re
from .normalization_dict import normalization_dict as norm_dict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords


class TextUtil:
    stop_words = stopwords.words('english')

    def __init__(self, normalization_dict=None):
        if normalization_dict is None:
            self._normalization_dict = norm_dict

    def text_normalization(self, text):
        """
        文本规划化 eg. it's -> it is， won't -> will not
        """
        for pattern, repl in self._normalization_dict.items():
            text = re.sub(pattern, repl, text)
        return text

    @classmethod
    def get_wordnet_pos(cls, treebank_tag):
        """
        分词词性格式转化为wordnet词性
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @classmethod
    def lemmatize_sentence(cls, sentence):
        """
        将句子分词并进行词性转化
        :param: sentence:
        :return: word array
        """
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = TextUtil.get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
        return res

    @classmethod
    def filter_stop_word(cls, words):
        return [word for word in words if word not in TextUtil.stop_words]

    @classmethod
    def filter_punctuation(cls, words):
        return [word for word in words if word.isalpha()]