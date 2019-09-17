import os
import torch
import pickle
import glob


def data_generator(data_dir, args):
    corpus_path = os.path.join(data_dir, 'corpus', 'corpus')
    if os.path.exists(corpus_path) and not args.corpus:
        # print(os.path.exists(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = Corpus(data_dir)
        pickle.dump(corpus, open(corpus_path, 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'':0}
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

        self.trainX, self.trainY = self.tokenize(os.path.join(path, 'train/'))
        self.testX, self.testY = self.tokenize(os.path.join(path, 'test/'))

    def tokenize(self, path):
        # fare un vettore con tutte le lunghezze delle frasi per
        assert os.path.exists(path)
        # Add words to the dictionary
        max_seq_len = 0
        n_file = 0
        data = []
        targets = []
        for filename in glob.glob(os.path.join(path,'*.txt')):
            file_number, target_with_path = filename.split('_')
            target = target_with_path.split('.')[0]
            target = int(target)
            with open(filename, 'r') as f:
                content = f.read()
                words = content.split()
                for word in words:
                    self.dictionary.add_word(word)

                ids = torch.LongTensor(len(words))
                token = 0

                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

                data.append(ids)
                targets.append(target)
        return data, targets


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size) #?????????????
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    #if args.cuda:
    #    data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]     # CAUTION: This is un-flattened! --> il +1 serve per il supervised
    return data, target