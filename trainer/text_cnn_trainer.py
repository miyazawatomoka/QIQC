from torch.autograd import Variable
from model import TextCNN
from torch.utils.data import DataLoader
from config import Config
from torch import optim
import torch
from utils import TrainerUtil


class TextCNNTrainer:
    def __init__(self, train_dataset, test_dataset):
        self.model = TextCNN()
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=Config.CNN_BATCH_SIZE)
        self.test_loader = DataLoader(test_dataset)
        # Adam优化器参数为默认,可调参
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = torch.nn.BCELoss()
        if Config.USE_GPU:
            self.model.cuda()

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            vdata = Variable(data)
            vlabel = Variable(label.float())
            if Config.USE_GPU:
                vdata = vdata.cuda()
                vlabel = vlabel.cuda()
            predict = self.model(vdata)
            loss = self.criterion(predict, vlabel)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        print('Average loss: {:.4f}'.format(train_loss / len(self.train_loader.dataset)))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct_count = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.test_loader):
                vdata = Variable(data)
                vlabel = Variable(label.float())
                if Config.USE_GPU:
                    vdata = vdata.cuda()
                    vlabel = vlabel.cuda()
                predict = self.model(vdata)
                correct_count += TrainerUtil.get_sigmoid_correct_count(predict, vlabel)
                loss = self.criterion(predict, vlabel)
                test_loss += loss.item()
        print("Total loss: {:.4f}".format(test_loss))
        print("Accuracy is: {: .4f}".format(correct_count / len(self.test_loader.dataset)))
