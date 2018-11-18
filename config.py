class Config:
    # GPU
    USE_GPU = False

    # CHAR
    PADDING_CHAR = '<pad>'
    START_CHAR = '<sos>'
    END_CHAR = '<eos>'
    UNKNOWN_CHAR = '<unk>'

    # PATH
    LABEL_PATH = '../data/train_label.npy'

    # WORD EMBEDDING PATH
    WORD2VEC_PATH = 'D:/dataset/word2vec/GoogleNews-vectors-negative300.bin'
    WORD2VEC_DATA_PATH = '../data/word2vec_martix.npy'

    # WORD EMBEDDING PARAMETER
    SENTENCE_MAX_LEN = 50
    EMBEDDING_SIZE = 300
    WORD_FREQUENCY = 3                              # 词频数大于这个列入词列表

    # TEXT CNN
    CNN_OUT_CHANNEL_NUM = 20
    CNN_DROPOUT = 0.5
    CNN_KERNEL_NUM = 100
    CNN_BATCH_SIZE = 20
    CNN_EPOCH = 10


    # NormTrainer
    Norm_BATCH_SIZE = 20
    # PARAMTER
    TESTSET_RATE = 0.3
    POSTIVE_TIMES = 4                                # 平均训练几次加入一个正样本