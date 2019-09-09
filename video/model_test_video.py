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
parser.add_argument('--cuda', action='store_true')
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
parser.add_argument('--model', type=str, default='m_data_pouring_d_400_h_4_type_GR.pt')
parser.add_argument('--test', type=str, default='test')

args = parser.parse_args()

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path, 'data/')
output_dir = os.path.join(base_path, 'output/')
image_dir = os.path.join(base_path, 'images/')

model_filename = os.path.join(output_dir, args.model)
if args.cuda:
    device = torch.device('cuda')
with open(model_filename, 'rb') as f:
    model = torch.load(f)
    if args.cuda:
        model.to(device)
model.eval()

test_dir = os.path.join(data_dir, args.test)
max_seq_len = 50
num_pixels = int(args.test.split('test')[1])


def sequence_forecast(test_sequence, number): #da n a 1
    images = []

    for i in range(number, 0, -1):

        x = test_sequence[:-i].float() / 255
        #print(x.shape)
        # per ora prendo come y tutto tranne l'ultima immagine --> poi posso prendere tutte tranne le ultime n-k immagini
        # e prevedere la n-k immagine, poi la n-k+1 e così via fino a n
        output = model(x.unsqueeze(0)).squeeze(0)
        output_image = (output[-1] )
        if (i != number):
            test_sequence[-i] = output_image

        images.append((output_image*255).detach().numpy())

    return images

def line2grid(line_image):
    img = np.zeros([28, 28])
    i = 0
    j = 0

    for element in line_image:
        img[i, j] = element

        if (j == 27):
            j = 0
            i += 1
        else:
            j += 1

    return Image.fromarray(np.uint8(img))

if __name__ == "__main__":


    X_test = get_data_list(test_dir, num_pixels, max_seq_len)
    X_train = get_data_list(test_dir, num_pixels, max_seq_len)

    # prendo la prima sequenza da testare --> prevo ultimo frame

    #test_sequence = X_test[0] # prova
    test_sequence = X_train[0]

    output_images = sequence_forecast(test_sequence, 4)

    #print((output_images))

    #ora voglio ricostruire l'immagine output

    for count, img in enumerate(output_images):
        path = os.path.join(image_dir, 'output_image' + str(count) + '.jpg')
        line2grid(img).save(path)

    #output_image = (output[-1]*255)
    #output_image = output_image.detach().numpy()



    #image = Image.fromarray(np.uint8(img))
    #image.show()

