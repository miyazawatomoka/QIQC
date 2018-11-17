import torch


class TrainerUtil:
    @staticmethod
    def get_sigmoid_correct_count(predict, tlabel):
        predict = torch.round(predict)
        diff = tlabel.float().sub(predict)
        abs = torch.abs(diff)
        return predict.size()[0] - torch.sum(abs).item()
