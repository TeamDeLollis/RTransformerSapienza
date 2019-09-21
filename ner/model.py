import torch
import torch.nn.functional as F
import sys
import os
from torch import nn
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, '../models'))
from RTransformer import RTransformer


class RT(nn.Module):
    def __init__(self,  dict_size, input_size, output_size, h, rnn_type, ksize, n_level, n, 
                 dropout=0.2, emb_dropout=0.2, cuda=False, emb_weights=None):

        super(RT, self).__init__()

        self.encoder, num_embeddings, embedding_dim = self.create_emb_layer(emb_weights)
        # self.encoder = nn.Embedding(dict_size, input_size)
        self.rt = RTransformer(embedding_dim, rnn_type, ksize, n_level, n, h, dropout, cuda)
        self.decoder = nn.Linear(embedding_dim, output_size)

        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # noinspection PyTypeChecker
        emb = self.drop(self.encoder(input))
        y = self.rt(emb)
        y = self.decoder(y)
        return F.log_softmax(y, dim=2)

    def create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim
