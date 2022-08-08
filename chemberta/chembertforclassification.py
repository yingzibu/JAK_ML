import torch
import torch.nn as nn
import torch.nn.functional as F
from const import *
from encoder import encoder
from bert_encoder import chembert_encoder


class chembertforclassification(nn.Module):
    def __init__(self, load_path='checkpoints/chem_bert_encoder_pretrain_9.pt',
                 last_layer_size=fps_len, output_size=2,dropout=0.5):
        super(chembertforclassification, self).__init__()

        self.last_layer_size = last_layer_size
        self.output_size = output_size
        self.pretrained = chembert_encoder()
        self.pretrained.load_state_dict(
            torch.load(load_path, map_location=device))

        self.w = nn.Linear(self.last_layer_size, self.output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.w(self.dropout(self.pretrained(x)))
