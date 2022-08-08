import torch
import torch.nn as nn
from const import *
from transformers import AutoModelWithLMHead, AutoTokenizer

class chembert_encoder(nn.Module):
    def __init__(self, output_dim=fps_len,dropout=0.5):
        super(chembert_encoder, self).__init__()
        self.bert = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.dropout=nn.Dropout(dropout)
        self.w=nn.Linear(767,output_dim)

    def forward(self, x):
        input_feat = self.tokenizer.batch_encode_plus(x, max_length=512,
                                                 padding='longest',  # implements dynamic padding
                                                 truncation=True,
                                                 return_tensors='pt',
                                                 return_attention_mask=True,
                                                 return_token_type_ids=True
                                                 )

        if cuda_available:
            input_feat['attention_mask'] = input_feat['attention_mask'].cuda()
            input_feat['input_ids'] = input_feat['input_ids'].cuda()


        outputs = self.bert(input_feat['input_ids'], attention_mask=input_feat['attention_mask'],output_hidden_states=None).logits[:,0,:]
        return self.w(self.dropout(outputs))
