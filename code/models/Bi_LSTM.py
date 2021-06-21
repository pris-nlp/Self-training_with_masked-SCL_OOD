"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

RNN_OUTPUT_DIM = 128
RNN_NUM_LAYERS = 1
RNN_HIDDEN_DIM = 512

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, HIDDEN_DIM=RNN_HIDDEN_DIM, NUM_LAYERS=RNN_NUM_LAYERS, OUTPUT_DIM=RNN_OUTPUT_DIM):
        super(LSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.output_dim = OUTPUT_DIM
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                      _weight=torch.from_numpy(embedding_matrix))
        self.rnn = nn.GRU(input_size=embedding_matrix.shape[1], hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)


    def forward(self, x):
        self.rnn.flatten_parameters()
        seq_embed = self.embedding(x)
        _, ht = self.rnn(seq_embed.to(dtype=torch.float32))
        ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
        output = self.fc(ht)
        return output


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='linear', feat_dim=128, embedding_matrix=None):
        super(SupConResNet, self).__init__()
        dim_in = RNN_OUTPUT_DIM
        self.encoder = LSTM(embedding_matrix)

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat_base = self.encoder(x) # 用于计算 lof
        logits = self.head(feat_base)
        feat = F.normalize(logits, dim=1) # 用于训练 scl
        probs = torch.softmax(feat, dim=1) # 用于分类指标计算
        return feat, logits, feat_base, probs
