import torch.nn.functional as F
import os, sys
from torch import nn
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from RTransformer import RTransformer
import torch


class RT(nn.Module):
    def __init__(self, d_model, input_size, num_classes, h, rnn_type, ksize, n_level, n, dropout=0.2, emb_dropout=0.2, tied_weights=False, cuda=False, max_len=0):
        super(RT, self).__init__()
        self.encoder = nn.Embedding(input_size, d_model) #input_size embedding with dimension d_model
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout, cuda)
        self.decoder = nn.Linear(d_model, num_classes)
        self.decoder_extra = nn.Linear(max_len,1)

        if tied_weights:
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)
        x = x.transpose(-2,-1)
        x = self.encoder(x)
        x = self.rt(x)  # input should have dimension (N, C, L)
        x = x.transpose(-2,-1)
        o = self.linear(x[:, :, -1])
        return F.log_softmax(o, dim=1)
        """
        #x = x.transpose(-2,-1)
        emb = self.drop(self.encoder(x))
        y = self.rt(emb)
        y = self.decoder(y).squeeze(0)
        y = y.transpose(1,0)
        #y = y.transpose(-2, -1)
        y = self.decoder_extra(y)
        y = y.transpose(0,1)
        #y =  y[: , -1, :]
        #return F.softmax(y, dim=1)
        return F.log_softmax(y, dim=1)



