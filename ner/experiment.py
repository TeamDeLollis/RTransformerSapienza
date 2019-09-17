import numpy as np
import argparse
import time
import math
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import RT

sys.path.append("../../")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.35)
parser.add_argument('--emb_dropout', type=float, default=0.15)
parser.add_argument('--clip', type=float, default=0.35)
parser.add_argument('--epochs', type=int, default=100)
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
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    device = torch.device("cuda")

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path, '{}/'.format(args.data))
s_dir = os.path.join(base_path, 'output/')

print(args)
print(data_dir)

corpus = data_generator(data_dir, args)
# eval_batch_size = 10

train_X = corpus.train_X
train_Y = corpus.train_Y
test_X = corpus.test_X
test_Y = corpus.test_Y

n_words = len(corpus.dictionary)
n_categories = len(corpus.categories)

dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied


model = RT(n_words, args.d_model, n_categories, h=args.h, rnn_type=args.rnn_type, ksize=args.ksize,
           n_level=args.n_level,  n=args.n, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied, cuda=args.cuda)

if args.cuda:
    model.to(device)

model_name = "d_{}_h_{}_type_{}_ks_{}_level_{}_n_{}_lr_{}_drop_{}".format(args.d_model, args.h, args.rnn_type,
                                                                          args.ksize, args.n_level, args.n, args.lr,
                                                                          args.dropout)
message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

# May use adaptive softmax to speed up training
criterion = nn.NLLLoss()  # CrossEntropyLoss()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def save(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def output_s(message, save_filename):
    print(message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')


def evaluate(data_X, data_Y):
    model.eval()
    total_loss = 0
    # processed_data_size = 0
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(data_X) - 1, args.batch_size)):
            # for i in range(0, data_source.size(1) - 1, args.validseqlen):
            # if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
            #   continue
            # data, targets = get_batch(data_source, i, args, evaluation=True)
            data, targets = get_batch(data_X, data_Y, args.batch_size, batch_idx) # args)
            if args.cuda:
                data, targets = data.cuda(), targets.cuda()
            output = model(data)

            # Discard the effective history, just like in training
            # eff_history = args.seq_len - args.validseqlen
            # final_output = output[:, eff_history:].contiguous().view(-1, n_words)
            # final_target = targets[:, eff_history:].contiguous().view(-1)

            loss = criterion(output.transpose(2, 1), targets)

            # Note that we don't add TAR loss here
            total_loss += loss.item()  # (data.size(1) - eff_history) * loss.item()
            # processed_data_size += data.size(1) - eff_history
        return total_loss  # / processed_data_size


def train():
    # Turn on training mode which enables dropout.
    global train_X
    global train_Y
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, len(train_X) - 1, args.batch_size)):  # args.validseqlen)):
        # if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
        #    continue
        data, targets = get_batch(train_X, train_Y, args.batch_size, batch_idx) # args)
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        optimizer.zero_grad()
        output = model(data)

        # Discard the effective history part
        # eff_history = args.seq_len - args.validseqlen
        # if eff_history < 0:
        #     raise ValueError("Valid sequence length must be smaller than sequence length!")
        # final_target = targets[:, eff_history:].contiguous().view(-1)
        # final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(output.transpose(2, 1), targets)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            message = ('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                       'loss {:5.6f} | ppl {:8.2f}'.format(
                        epoch, batch_idx,  len(train_X) // args.batch_size, lr,
                        elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            output_s(message, message_filename)
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(test_X, test_Y)
            # test_loss = evaluate(test_X, test_Y)
            test_loss = val_loss
            message = ('-' * 89
                       + '\n| end of epoch {:3d} | time: {:5.6f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss))
                       + '\n| end of epoch {:3d} | time: {:5.6f}s | test loss {:5.2f} | '
                       'test ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time),
                                                   test_loss, math.exp(test_loss))
                       + '-' * 89)
            output_s(message, message_filename)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                save(model, model_filename)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(model_filename, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_X, test_Y)
    message = ('=' * 89
               + '\n| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss))
               + "\n" + '=' * 89)
    output_s(message, message_filename)
