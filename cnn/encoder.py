import torch
import torch.nn as nn
from const import *


class encoder(nn.Module):
    def __init__(self, input_length, num_words, embedding_size=32, inner_size=32, output_size=fps_len, stride=1):
        super(encoder, self).__init__()

        self.input_length = input_length
        self.num_words = num_words
        self.embedding_size = embedding_size
        self.inner_size = inner_size
        self.output_size = output_size
        self.stride = stride

        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

        self.conv_1 = nn.Conv1d(self.embedding_size, self.inner_size, 1, self.stride)
        self.conv_2 = nn.Conv1d(self.embedding_size, self.inner_size, 2, self.stride)
        self.conv_3 = nn.Conv1d(self.embedding_size, self.inner_size, 3, self.stride)

        self.w = nn.Linear(self.inner_size * 3, self.output_size)

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.conv_1.weight)
        torch.nn.init.xavier_uniform_(self.conv_2.weight)
        torch.nn.init.xavier_uniform_(self.conv_3.weight)
        torch.nn.init.xavier_uniform_(self.w.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        tri = self.conv_3(x)
        bi = self.conv_2(x)
        uni = self.conv_1(x)

        tri_maxpool = nn.MaxPool1d(tri.shape[2])
        bi_maxpool = nn.MaxPool1d(bi.shape[2])
        uni_maxpool = nn.MaxPool1d(uni.shape[2])
        integrate_feat = torch.cat(
            (tri_maxpool(tri).squeeze(2), bi_maxpool(bi).squeeze(2), uni_maxpool(uni).squeeze(2)), dim=1)
        #print(integrate_feat.shape)
        return self.w(self.activation(integrate_feat))
