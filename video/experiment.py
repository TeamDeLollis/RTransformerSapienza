from utils import *
from model import RT
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
import os
import warnings

sys.path.append("../../")

warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clip', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ksize', type=int, default=6)  # window --> we should keep it small maybe 4
parser.add_argument('--n_level', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--rnn_type', type=str, default='GRU')  # LSTM; RNN; RNN_TANH; RNN_RELU
parser.add_argument('--d_model', type=int, default=400)  # da provare piu grande
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=4)  # we shouldn't need so many because no need for long term memory
parser.add_argument('--seed', type=int, default=44)
parser.add_argument('--data', type=str, default='pouring')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda')

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path, 'data/')
train_dir = os.path.join(data_dir, 'newTrain/')
test_dir = os.path.join(data_dir, 'newTest/')
s_dir = os.path.join(base_path, 'output/')

num_seq_train = 11
num_seq_test = 10
max_seq_len = 50
num_pixels = 28
input_size = 784  # 28 * 28
# X_train = get_data(train_dir, num_pixels, num_seq_train, max_seq_len)
# X_test = get_data(test_dir, num_pixels, num_seq_test, max_seq_len)
X_train = get_data_list(train_dir, num_pixels, max_seq_len)
X_test = get_data_list(test_dir, num_pixels, max_seq_len)

dropout = args.dropout
emb_dropout = args.dropout

model = RT(input_size, args.d_model, input_size, h=args.h, rnn_type=args.rnn_type,
           ksize=args.ksize, n_level=args.n_level, n=args.n, dropout=dropout, emb_dropout=emb_dropout)
model.to(device)

model_name = "data_{}_d_{}_h_{}_type_{}_k_{}_level_{}_n_{}_lr_{}_drop_{}".format(args.data, args.d_model, args.h,
                                                                                 args.rnn_type, args.ksize,
                                                                                 args.n_level, args.n, args.lr,
                                                                                 args.dropout)

message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def save(save_model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(save_model, f)
    print('Saved as %s' % save_model)


def output_s(output_message, save_filename):
    print(output_message)
    with open(save_filename, 'a') as file:
        file.write(output_message + '\n')


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0

    with torch.no_grad():  # no training
        for idx in eval_idx_list:
            data_line = X_data[idx]
            # print(data_line.shape)
            x, y = data_line[:-1].float(), data_line[1:].float()  # everything except first word, y = everything to all
            output = model(x.unsqueeze(0)).squeeze(0)

            #loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
            #                    torch.matmul((1 - y), torch.log(1 - output).float().t()))  # negative likelihood
            loss = #output-input
            total_loss += loss.item()

            count += output.size(0)
        eval_loss = total_loss / count
        message = name + " loss: {:.5f}".format(eval_loss)
        output_s(message, message_filename)
        return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = data_line[:-1].float(), data_line[1:].float()  # da prendere fino all'ultima prima di zero
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)

        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1 - y), torch.log(1 - output).float().t()))

        total_loss += loss.item()
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            message = "Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss)
            output_s(message, message_filename)
            total_loss = 0.0
            count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    for ep in range(1, args.epochs + 1):
        train(ep)
        vloss = evaluate(X_test, name='Validation')
        if vloss < best_vloss:
            save(model, model_filename)
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            output_s('lr = {}'.format(lr), message_filename)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)

    message = '-' * 89
    output_s(message, message_filename)
    model = torch.load(open(model_filename, "rb"))
    tloss = evaluate(X_test)
