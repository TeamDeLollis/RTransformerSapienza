import argparse
import torch.nn as nn
import torch.optim as optim
import sys, os, time, math, warnings
sys.path.append("../../")
from utils import *
from model import RT


if __name__ == "__main__":

    warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ksize', type=int, default=6) #finestra
    parser.add_argument('--n_level', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--data', type=str, default='Nott')


    args = parser.parse_args()

    #model_filename = "/output/m_data_Nott_d_160_h_4_type_GRU_k_6_level_3_n_1_lr_5e-05_drop_0.1.pt"
    model_filename = "/home/lorenz/Desktop/Neural Networks/project/R-transformer-master/audio/output/m_data_Nott_d_160_h_4_type_GRU_k_6_level_3_n_1_lr_5e-05_drop_0.1.pt"
    with open(model_filename, 'rb') as f:
        model = torch.load(f)

    model.eval()

    base_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_path,'data/')
    X_train, X_valid, X_test = data_generator(args.data, data_dir)


    data_line = X_valid[6] #riga 6

    x, y = data_line[:-1], data_line[1:] #tutto tranne prima parola, y = tutto tranne ulima
    output = model(x.unsqueeze(0)).squeeze(0)

    print("target" ,y.shape)
    print("output" ,output)
#confronto tra y e output

    for count1, i in enumerate(output):
        for count2, j in enumerate(i):
            if (j > 10^-1):
                print("target", y[count1,count2])
                print("output", j)