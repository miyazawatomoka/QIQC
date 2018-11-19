import sys
sys.path.append('../')
from trainer import NormTrainer
from config import Config
from dataset import Word2vecStaticDataset
from model import TextCNN
from utils import Word2vecUtil, WordEmbeddingUtil
import gc
import torch

train_dataset = Word2vecStaticDataset(is_train=True,
                                           label_path='../data/train_label.npy',
                                           data_path='../data/word2vec_idx_data.npy')
test_dataset = Word2vecStaticDataset(is_train=False,
                                          label_path='../data/train_label.npy',
                                          data_path='../data/word2vec_idx_data.npy')


def main():
    print("Get pre-trained embedding weight...")
    word2vec_util = Word2vecUtil(word2vec_path=Config.WORD2VEC_PATH)
    wordembedding_util = WordEmbeddingUtil()
    pre_weight = word2vec_util.get_weight()
    emb_weight = wordembedding_util.get_embedding_weight(pre_weight)
    emb_weight = torch.tensor(emb_weight)
    gc.collect()
    print("Get pre-trained embedding weight finished")
    print("Build model...")
    model = TextCNN(pretrained_weight=emb_weight, is_static=False).double()
    print("Build model finished")
    trainer = NormTrainer(model=model, train_dataset=test_dataset, test_dataset=test_dataset)
    print("Begin Train Text-CNN")
    for epoch in range(Config.CNN_EPOCH):
        print("===============================  EPOCH {:d}  ===============================".format(epoch))
        trainer.train()
        if epoch % 5 == 0:
            print("===============================  Test  ===============================")
            trainer.test()
    trainer.save_model('../pretrained/text_cnn_static.h5')


if __name__ == '__main__':
    main()
