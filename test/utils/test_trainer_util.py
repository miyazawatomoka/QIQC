from unittest import TestCase
from utils import TrainerUtil
import torch


class TestTrainUtil(TestCase):
    def test_get_sigmoid_correct_count(self):
        lab1 = torch.tensor([1,0,1])
        pre1 = torch.tensor([0.7, 0.3, 0.8])
        out1 = TrainerUtil.get_sigmoid_correct_count(pre1, lab1)
        expect1 = 3
        self.assertLessEqual(abs(out1 - expect1), 0.001)
