import pickle
import bcolz

glove_path = './data'

vectors = bcolz.open(f'{glove_path}/6B.100.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
