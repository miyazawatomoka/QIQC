from torch.utils import data
from config import Config
import numpy as np
from sklearn.model_selection import train_test_split
import torch


class Word2vecStaticDataset(data.Dataset):
    def __init__(self, is_train=False, label_path=Config.LABEL_PATH, data_path=Config.WORD2VEC_DATA_PATH):
        self.is_train = is_train
        self.label = np.load(label_path)
        self.data = np.load(data_path)
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(self.data,
                                                                                              self.label,
                                                                                              test_size=Config.TESTSET_RATE)
        self.train_data_postive_idxs = np.array(np.where(self.label_train == 1)).ravel()

    def __getitem__(self, index):
        if self.is_train:
            if np.random.randint(Config.POSTIVE_TIMES) == 0:
                idx = np.random.choice(self.train_data_postive_idxs)
                return torch.LongTensor(self.data_train[idx]), self.label_train[idx]
            else:
                return torch.LongTensor(self.data_train[index]), self.label_train[index]
        else:
            return torch.LongTensor(self.data_test[index]), self.label_test[index]

    def __len__(self):
        if self.is_train:
            return len(self.label_train)
        else:
            return len(self.label_test)
