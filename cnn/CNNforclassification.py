import torch
import torch.nn as nn
import torch.nn.functional as F
from const import *
from encoder import encoder


class CNNforclassification(nn.Module):
    def __init__(self, max_len, voc_len, load_path='checkpoints/CNN_encoder_pretrain2.pt',
                 last_layer_size=fps_len, output_size=2):
        super(CNNforclassification, self).__init__()

        self.last_layer_size = last_layer_size
        self.output_size = output_size
        self.pretrained = encoder(max_len, voc_len).cuda()
        self.pretrained.load_state_dict(
            torch.load(load_path, map_location=device))

        self.w = nn.Linear(self.last_layer_size, self.output_size)

        self.activation = nn.LeakyReLU()

    def forward(self, x):

        return self.w(self.activation(self.pretrained(x)))
