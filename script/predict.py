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
df = pd.read_csv('../../input/test.csv')
model = torch.load('../../pretrained/text_cnn_static.h5')
tdata = np.load('../../cache/word2vec_idx_test.npy')

def cnn_predict():

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
