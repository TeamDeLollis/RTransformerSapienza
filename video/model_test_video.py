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
from PIL import Image


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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--rnn_type', type=str, default='GRU')  # LSTM; RNN; RNN_TANH; RNN_RELU
parser.add_argument('--d_model', type=int, default=400)  # da provare piu grande
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=4)  # we shouldn't need so many because no need for long term memory
parser.add_argument('--seed', type=int, default=44)
parser.add_argument('--data', type=str, default='pouring')

args = parser.parse_args()

    #model_filename = "/output/m_data_Nott_d_160_h_4_type_GRU_k_6_level_3_n_1_lr_5e-05_drop_0.1.pt"
model_filename = "/home/lorenz/Desktop/Neural Networks/project/R-transformer-master/video/output/m_data_pouring_d_400_h_4_type_GRU_k_6_level_3_n_1_lr_0.001_drop_0.2.pt"
with open(model_filename, 'rb') as f:
    model = torch.load(f)

model.eval()

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path,'data/')
test_dir = os.path.join(data_dir, 'newTest/')
max_seq_len = 50
num_pixels = 28

X_test = get_data_list(test_dir, num_pixels, max_seq_len)

# prendo la prima sequenza da testare --> prevo ultimo frame

test_sequence = X_test[0]

x, y = test_sequence[:-1].float()/255, test_sequence[1:].float()/255
# per ora prendo come y tutto tranne l'ultima immagine --> poi posso prendere tutte tranne le ultime n-k immagini
# e prevedere la n-k immagine, poi la n-k+1 e così via fino a n

output = model(x.unsqueeze(0)).squeeze(0)

#ora voglio ricostruire l'immagine output

output_image = (output[-1]*255)
output_image = output_image.detach().numpy()

img = np.zeros([28, 28])
i = 0
j = 0

for element in output_image:
    print(j,i)

    img[i,j] = element

    if (j == 27):
        j = 0
        i += 1
    else:
        j +=1

image = Image.fromarray(np.uint8(img))
image.show()

