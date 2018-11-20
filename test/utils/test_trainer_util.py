from unittest import TestCase
from utils import TrainerUtil
import torch
import numpy as np


class TestTrainUtil(TestCase):
    def test_get_sigmoid_correct_count(self):
        lab1 = torch.FloatTensor([1,0,1])
        pre1 = torch.FloatTensor([0.7, 0.3, 0.8])
        out1 = TrainerUtil.get_sigmoid_correct_count(pre1, lab1)
        expect1 = 3
        self.assertLessEqual(abs(out1 - expect1), 0.001)

    def test_get_f1_score_by_predict_sigmoid(self):
        lab1 = np.array([1, 0, 0])
        pre1 = np.array([0.7, 0.1, 0.2])
        x = TrainerUtil.get_f1_score_by_predict_sigmoid(pre1, lab1)
        self.assertEqual(x, 1.0)
