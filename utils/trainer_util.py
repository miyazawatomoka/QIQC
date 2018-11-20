import torch
from sklearn.metrics import f1_score
import numpy as np


class TrainerUtil:
    @staticmethod
    def get_sigmoid_correct_count(predict, tlabel):
        predict = torch.round(predict)
        diff = tlabel.sub(predict)
        abs = torch.abs(diff)
        return predict.size()[0] - torch.sum(abs).item()

    @staticmethod
    def get_f1_score_by_predict_sigmoid(predict, true_lable):
        predict = np.round(predict)
        true_lable = np.round(true_lable)
        return f1_score(predict, true_lable)
