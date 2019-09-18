import argparse
import time, math, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append("../../")
from utils import *
from model import RT
import pickle
from random import randint
import numpy as np

if __name__ == "__main__":

    model_filename = "/home/lorenz/Desktop/Neural Networks/project/R-transformer-master/sentiment/output/r_d_128_h_8_t_LSTM_ksize_9_level_3_n_1_lr_0.001_dropout_0.25.pt"
    with open(model_filename, 'rb') as f:
        model = torch.load(f)
    model.eval()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--emb_dropout', type=float, default=0.15)
    parser.add_argument('--clip', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ksize', type=int, default=9)
    parser.add_argument('--n_level', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=200, metavar='N')
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--validseqlen', type=int, default=40)
    parser.add_argument('--seq_len', type=int, default=80)
    parser.add_argument('--tied', action='store_false')
    parser.add_argument('--data', type=str, default='penn')
    parser.add_argument('--corpus', action='store_true',
                        help='force re-make the corpus (default: False)')
    args = parser.parse_args()
    base_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_path, 'data/{}/'.format(args.data))
    corpus = data_generator(data_dir, args)

    train_data = corpus.trainX
    train_y = corpus.trainY
    test_data = corpus.testX
    test_y = corpus.testY

    targets_N = test_data[0]
    output = model(targets_N.unsqueeze(0))
    pred = output.data.max(1, keepdim=True)[1]

    print("recensione presa dal test: ", targets_N)
    print("previsione: ", pred + 1)
    print("target: ", test_y[0])


