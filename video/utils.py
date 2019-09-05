from scipy.io import loadmat
import torch
import numpy as np
from PIL import Image
import glob

def data_generator(dataset, data_dir):
    print('loading Nott data...')
    data = loadmat(data_dir + 'Nottingham.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        print (dataset, len(data))
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test


def get_data(number_of_sequences, max_sequence_length, number_of_pixels):

	data = np.zeros([number_of_sequences, max_sequence_length, number_of_pixels, number_of_pixels])
	"""for j in range(0,num_sequences):
		train[j] = np.zeros(max_sequence_length)
		print(train[j])
		for k in range(0,max_sequence_length):
			train[j][k] = np.zeros([500,500])
	"""
	images = glob.glob("*.jpg")
	for image in images:
		actual_seq = image[8:11]
		index_n = image[12:15]
		img = Image.open(image)     #Open an Image via PILLOW
		img = img.convert('L')      #Grayscale transformation
		imVect = np.array(img)
		actual_seq = int(actual_seq) - 1
		index_n = int(index_n) - 1
		print(data[0])

		for i in range(0,500):
			for j in range(0,500):
				data[actual_seq][index_n][i][j] = imVect[i][j]



	data = torch.from_numpy(data)
	data.view(1,2,500*500)

	return data