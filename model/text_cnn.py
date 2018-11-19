import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config


class TextCNN(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, pretrained_weight):
        super(TextCNN, self).__init__()
        self.with_embedding = False
        in_channel = 1
        out_channel = Config.CNN_KERNEL_NUM
        kernel_sizes = [3, 4, 5]
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight, freeze=False)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, Config.EMBEDDING_SIZE)) for K in kernel_sizes])

        self.dropout = nn.Dropout(Config.CNN_DROPOUT)
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        x = self.embedding(input_x)

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        # logit = F.log_softmax(self.fc(x))  # (batch_size, num_aspects)
        fc = self.fc(x)
        sigmoid = self.sigmoid(fc)
        return sigmoid
