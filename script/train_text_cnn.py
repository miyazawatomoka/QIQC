import sys
sys.path.append('../')
from trainer import NormTrainer
from config import Config
from dataset import Word2vecStaticDataset
from model import TextCNN

train_dataset = Word2vecStaticDataset(is_train=True,
                                           label_path='../data/train_label.npy',
                                           data_path='../data/word2vec_martix.npy')
test_dataset = Word2vecStaticDataset(is_train=False,
                                          label_path='../data/train_label.npy',
                                          data_path='../data/word2vec_martix.npy')


def main():
    model = TextCNN()
    trainer = NormTrainer(model=model, train_dataset=test_dataset, test_dataset=test_dataset)
    for epoch in range(Config.CNN_EPOCH):
        print("===============================  EPOCH {:d}  ===============================".format(epoch))
        trainer.train()
        trainer.test()
    trainer.save_model('../pretrained/text_cnn_static.h5')


if __name__ == '__main__':
    main()
