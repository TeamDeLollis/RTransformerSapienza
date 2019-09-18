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
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--emb_dropout', type=float, default=0.15)
parser.add_argument('--clip', type=float, default=0.35)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--ksize', type=int, default=9)
parser.add_argument('--n_level', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=2500, metavar='N')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--rnn_type', type=str, default='LSTM') # o gru?
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=8)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--permute', action='store_true', default=False)
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
parser.add_argument('--tied', action='store_false')


args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path,'data/')
print(data_dir)
s_dir = os.path.join(base_path,'output/')

corpus = data_generator(data_dir, args)

train_data = corpus.trainX
train_y = corpus.trainY
test_data = corpus.testX
test_y = corpus.testY

eval_batch_size = 10

#train_data = batchify(corpus.train, args.batch_size, args)
#val_data = batchify(corpus.valid, eval_batch_size, args)
#test_data = batchify(corpus.test, eval_batch_size, args)

n_words = len(corpus.dictionary)

batch_size = args.batch_size
n_classes = 10
input_channels = 1
epochs = args.epochs
steps = 0
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied

model = RT(128, n_words, n_classes, h=args.h, rnn_type=args.rnn_type, ksize=args.ksize,
    n_level=args.n_level, n=args.n, dropout=args.dropout, emb_dropout=args.dropout)

model.to(device)

model_name = "d_{}_h_{}_t_{}_ksize_{}_level_{}_n_{}_lr_{}_dropout_{}".format(
            args.d_model, args.h, args.rnn_type, args.ksize, 
            args.n_level, args.n, args.lr, args.dropout)

message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def save(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')

def train(ep):
    global steps
    train_loss = 0
    model.train()
    batch_idx = 0
    steps = 0
    sequence = np.arange(len(train_data)) #in modo che siano presi in modo casuale
    np.random.shuffle(sequence)
    for index in sequence:
        data = train_data[index]
        target = train_y[index]
        if args.cuda: data = data.cuda()
        optimizer.zero_grad()
        output = model(data.unsqueeze(0))
        #print(torch.tensor([target]))
        #print(output)
        loss = F.nll_loss(output, torch.tensor([target - 1]).cuda())
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += 1
        if steps > 0 and steps % args.log_interval == 0:
            pred = output.data.max(1, keepdim=True)[1]
            message = ('Train Epoch: {} Loss: {:.6f}\tSteps: {} \tpred: {}  target: {}'.format(
                ep, train_loss.item()/args.log_interval, steps, pred.item() + 1, target))
            output_s(message, message_filename)
            train_loss = 0
        batch_idx += 1

def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_dim = 500
    with torch.no_grad():
        sequence = np.random.randint(0, high=len(test_data), size=test_dim) #test on 500 elements from the test set
        for index in sequence:
            data = test_data[index]
            target = test_y[index]
            if args.cuda:
                data = data.cuda()
            output = model(data.unsqueeze(0))
            loss = F.nll_loss(output, torch.tensor([target - 1]).cuda())
            pred = output.data.max(1, keepdim=True)[1]
            if (pred.item() + 1) == target:
                correct += 1
            break
            test_loss += loss

        message = ('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss.item()/test_dim, correct, test_dim,
            100. * correct / test_dim))
        output_s(message, message_filename)
        return test_loss


if __name__ == "__main__":

    for epoch in range(1, epochs+1):
        train(epoch)
        save(model, model_filename)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

