class Config:
    # CHAR
    PADDING_CHAR = '<pad>'
    START_CHAR = '<sos>'
    END_CHAR = '<eos>'
    UNKNOWN_CHAR = '<unk>'

    # WORD EMBEDDING PATH
    WORD2VEC_PATH = 'D:/dataset/word2vec/GoogleNews-vectors-negative300.bin'

    # WORD EMBEDDING PARAMETER
    SENTENCE_MAX_LEN = 50
    EMBEDDING_SIZE = 300
    WORD_FREQUENCY = 3                              # 词频数大于这个列入词列表

