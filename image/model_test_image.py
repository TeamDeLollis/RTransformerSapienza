import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append("../../")
from utils import data_generator
from model import RT
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--ksize', type=int, default=7)
parser.add_argument('--n_level', type=int, default=8)
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--rnn_type', type=str, default='GRU')
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--h', type=int, default=2)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--permute', action='store_true', default=False)
args = parser.parse_args()

model_filename = "/output/queoCHExe.pt"
    with open(model_filename, 'rb') as f:
        model = torch.load(f)

model.eval()

base_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(base_path,'data')
train_loader, test_loader = data_generator(root, args.batch_size)

data_N = test_loader[0][0]
target_N = test_loader[0][1]

output = model(data)
