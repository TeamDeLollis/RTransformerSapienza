import os
import torch
import pickle


def data_generator(data_dir, args):
    corpus_path = os.path.join(data_dir, 'corpus', 'corpus')
    if os.path.exists(corpus_path) and not args.corpus:
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = Corpus(data_dir)
        pickle.dump(corpus, open(corpus_path, 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'': 0}
        self.idx2word = ['']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.categories = {'I-PER': 0,
                           'I-LOC': 1,
                           'I-ORG': 2,
                           'I-MISC': 3,
                           'B-PER': 4,
                           'B-LOC': 5,
                           'B-ORG': 6,
                           'B-MISC': 7,
                           'O': 8}
        self.train_X, self.train_Y = self.tokenize(os.path.join(path, 'eng.train'))
        self.test_X, self.test_Y = self.tokenize(os.path.join(path, 'eng.testa'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        X, Y = [], []
        # Add words to the dictionary
        with open(path, 'r') as f:
            # tokens = 0
            for line in f:
                if line != '\n':
                    split = line.split(' ')
                    if split[0] != '-DOCSTART-':
                        word = split[0]
                        # print(word)
                        self.dictionary.add_word(word)

        # Tokenize file content
        line_num = 0
        skip = False
        X.append([])
        Y.append([])
        with open(path, 'r') as f:
            for line in f:
                # print(line)
                if line != '\n':
                    split = line.split(' ')
                    if split[0] != '-DOCSTART-':
                        skip = False
                        word = split[0]
                        # print(word)
                        target = split[-1].rstrip()
                        # print(target)
                        X[line_num].append(self.dictionary.word2idx[word])
                        Y[line_num].append(self.categories[target])
                    else:
                        skip = True
                elif not skip:
                    # print('*'*1 )
                    X[line_num] = torch.tensor(X[line_num])
                    Y[line_num] = torch.tensor(Y[line_num])
                    X.append([])
                    Y.append([])
                    line_num += 1
        X.pop()
        Y.pop()
        return X, Y


def get_batch(train, test, batch_size, i):  # args, seq_len=None, evaluation=False):
    num_seq = min(batch_size, len(train) - 1 - i * batch_size)
    X = train[i * batch_size: i * batch_size + num_seq]
    Y = test[i * batch_size: i * batch_size + num_seq]
    # print(len(Y))

    max_len = max(*[s.size() for s in X])
    # num_cat = max(*[c for y in Y for c in y]) + 1
    torchX = torch.zeros(batch_size, max_len, dtype=torch.long)
    torchY = torch.zeros(batch_size, max_len, dtype=torch.long) + 8  # VERY IMPORTANT to add 8
    for j in range(len(X)):
        torchX[j, :len(X[j])] = X[j]
        torchY[j, :len(Y[j])] = Y[j]
        # for k in range(len(Y[j]):
        #    torchY[j, k] = one_hot(Y[j][k])
    return torchX, torchY


# def batchify(data, batch_size, args):
#    """The output should have size [L x batch_size], where L could be a long sequence length"""
#    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
#    nbatch = data.size(0) // batch_size
#    # Trim off any extra elements that wouldn't cleanly fit (remainders).
#    data = data.narrow(0, 0, nbatch * batch_size)
#    # Evenly divide the data across the batch_size batches.
#    data = data.view(batch_size, -1)
#    if args.cuda:
#        data = data.cuda()
#    return data
